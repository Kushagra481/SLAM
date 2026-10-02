import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from geometry import build_covariance_3d, quat_to_rotmat

class PyTorchGaussianRasterizer(nn.Module):
    """
    Vectorized, fully differentiable PyTorch 3D Gaussian Splatting rasterizer.
    Supports camera pose gradient computation via autograd.
    """
    def __init__(self, image_height: int = 64, image_width: int = 64, fx: float = 50.0, fy: float = 50.0):
        super().__init__()
        self.H = image_height
        self.W = image_width
        self.fx = fx
        self.fy = fy
        self.cx = image_width / 2.0
        self.cy = image_height / 2.0

        # Pixel grid setup
        grid_v, grid_u = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32),
            torch.arange(self.W, dtype=torch.float32),
            indexing="ij"
        )
        self.register_buffer("grid_u", grid_u)
        self.register_buffer("grid_v", grid_v)

    def forward(
        self, 
        xyz: torch.Tensor, 
        scales: torch.Tensor, 
        quats: torch.Tensor, 
        opacities: torch.Tensor, 
        colors: torch.Tensor, 
        pose_cw: torch.Tensor,
        z_near: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Renders 3D Gaussians given camera pose T_cw.
        
        xyz: (N, 3) world coordinates
        scales: (N, 3) log scale parameters
        quats: (N, 4) quaternion rotations
        opacities: (N, 1) opacity values in [0, 1]
        colors: (N, 3) RGB colors in [0, 1]
        pose_cw: (4, 4) SE(3) transformation world to camera frame
        
        Returns:
            rendered_image: (3, H, W) tensor
            accumulated_alpha: (1, H, W) tensor
        """
        device = xyz.device
        xyz = xyz.to(torch.float32)
        scales = scales.to(torch.float32)
        quats = quats.to(torch.float32)
        opacities = opacities.to(torch.float32)
        colors = colors.to(torch.float32)
        pose_cw = pose_cw.to(torch.float32)
        
        N = xyz.shape[0]

        if N == 0:
            rendered = torch.zeros((3, self.H, self.W), device=device)
            alpha = torch.zeros((1, self.H, self.W), device=device)
            return rendered, alpha

        # 1. Transform world mean to camera coordinate frame
        R_cw = pose_cw[:3, :3]
        t_cw = pose_cw[:3, 3]
        xyz_cam = torch.mm(xyz, R_cw.T) + t_cw.unsqueeze(0) # (N, 3)

        x = xyz_cam[:, 0]
        y = xyz_cam[:, 1]
        z = xyz_cam[:, 2]

        # Depth filtering
        valid_depth = z > z_near
        if not valid_depth.any():
            return torch.zeros((3, self.H, self.W), device=device), torch.zeros((1, self.H, self.W), device=device)

        # 2. Compute 2D Screen Projection Centers
        u0 = self.fx * (x / z) + self.cx
        v0 = self.fy * (y / z) + self.cy

        # 3. Project 3D Covariance to 2D
        Sigma_3d = build_covariance_3d(scales, quats) # (N, 3, 3)
        
        # Transform covariance to camera frame: R_cw @ Sigma_3d @ R_cw^T
        R_cw_batch = R_cw.unsqueeze(0).expand(N, -1, -1)
        Sigma_cam = torch.bmm(R_cw_batch, torch.bmm(Sigma_3d, R_cw_batch.transpose(1, 2)))

        # Projection Jacobian J_proj (N, 2, 3)
        inv_z = 1.0 / z
        inv_z2 = inv_z ** 2
        
        J_proj = torch.zeros((N, 2, 3), device=device)
        J_proj[:, 0, 0] = self.fx * inv_z
        J_proj[:, 0, 2] = -self.fx * x * inv_z2
        J_proj[:, 1, 1] = self.fy * inv_z
        J_proj[:, 1, 2] = -self.fy * y * inv_z2

        # 2D Covariance: J @ Sigma_cam @ J^T + low-pass filter
        Sigma_2d = torch.bmm(J_proj, torch.bmm(Sigma_cam, J_proj.transpose(1, 2))) # (N, 2, 2)
        
        # Add 0.3px low-pass filter smoothing for numerical stability
        Sigma_2d[:, 0, 0] += 0.3
        Sigma_2d[:, 1, 1] += 0.3

        # Invert 2D Covariance
        det = Sigma_2d[:, 0, 0] * Sigma_2d[:, 1, 1] - Sigma_2d[:, 0, 1] * Sigma_2d[:, 1, 0]
        det = torch.clamp(det, min=1e-5)
        
        inv_Sigma_2d = torch.zeros_like(Sigma_2d)
        inv_Sigma_2d[:, 0, 0] = Sigma_2d[:, 1, 1] / det
        inv_Sigma_2d[:, 1, 1] = Sigma_2d[:, 0, 0] / det
        inv_Sigma_2d[:, 0, 1] = -Sigma_2d[:, 0, 1] / det
        inv_Sigma_2d[:, 1, 0] = -Sigma_2d[:, 1, 0] / det

        # 4. Sort Gaussians by front-to-back depth (z)
        depth_order = torch.argsort(z)
        
        # 5. Evaluate pixel intensities
        # Reshape grid: (1, H, W)
        u_grid = self.grid_u.to(device).unsqueeze(0) # (1, H, W)
        v_grid = self.grid_v.to(device).unsqueeze(0) # (1, H, W)

        # Distances to projected center for each Gaussian
        # u0, v0 shape (N, 1, 1)
        du = u_grid - u0.view(N, 1, 1) # (N, H, W)
        dv = v_grid - v0.view(N, 1, 1) # (N, H, W)

        # Quad form: du^2 * inv00 + 2*du*dv*inv01 + dv^2 * inv11
        quad_form = 0.5 * (
            du**2 * inv_Sigma_2d[:, 0, 0].view(N, 1, 1) +
            2 * du * dv * inv_Sigma_2d[:, 0, 1].view(N, 1, 1) +
            dv**2 * inv_Sigma_2d[:, 1, 1].view(N, 1, 1)
        )

        gaussian_weight = torch.exp(-torch.clamp(quad_form, min=0.0, max=10.0)) # (N, H, W)
        
        # Effective alpha per pixel: alpha_i * gaussian_weight
        alpha = opacities.view(N, 1, 1) * gaussian_weight # (N, H, W)
        alpha = torch.clamp(alpha, min=0.0, max=0.99)

        # Front-to-back alpha blending
        # Transmittance T_i = prod_{j < i} (1 - alpha_j)
        rendered_image = torch.zeros((3, self.H, self.W), device=device)
        accumulated_alpha = torch.zeros((1, self.H, self.W), device=device)
        transmittance = torch.ones((1, self.H, self.W), device=device)

        # Iterate sorted depth front-to-back
        for idx in depth_order:
            if not valid_depth[idx]:
                continue
            
            a = alpha[idx:idx+1] # (1, H, W)
            c = colors[idx].view(3, 1, 1) # (3, 1, 1)

            weight = a * transmittance # (1, H, W)
            rendered_image = rendered_image + c * weight
            accumulated_alpha = accumulated_alpha + weight
            transmittance = transmittance * (1.0 - a)

        return rendered_image, accumulated_alpha
