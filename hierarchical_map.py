import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from rasterizer import PyTorchGaussianRasterizer

class GaussianTier(nn.Module):
    """
    Storage for a single tier of 3D Gaussians with active masking and Fisher score tracking.
    """
    def __init__(self, capacity: int = 10000, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.capacity = capacity
        self.device = device
        
        self.xyz = nn.Parameter(torch.zeros((capacity, 3), device=device))
        self.scales = nn.Parameter(torch.zeros((capacity, 3), device=device)) # log scales
        self.quats = nn.Parameter(torch.zeros((capacity, 4), device=device))  # (w, x, y, z)
        self.opacities = nn.Parameter(torch.zeros((capacity, 1), device=device)) # sigmoid opacities
        self.colors = nn.Parameter(torch.zeros((capacity, 3), device=device)) # sigmoid colors

        # Metadata
        self.register_buffer("fisher_scores", torch.zeros(capacity, device=device))
        self.register_buffer("active_mask", torch.zeros(capacity, dtype=torch.bool, device=device))
        self.register_buffer("descriptors", torch.zeros((capacity, 32), device=device)) # Tier 0 keyframe descriptors

    @property
    def num_active(self) -> int:
        return int(self.active_mask.sum().item())

    def get_active_dict(self) -> Dict[str, torch.Tensor]:
        mask = self.active_mask
        return {
            "xyz": self.xyz[mask],
            "scales": self.scales[mask],
            "quats": self.quats[mask],
            "opacities": torch.sigmoid(self.opacities[mask]),
            "colors": torch.sigmoid(self.colors[mask])
        }


class HierarchicalGaussianMap(nn.Module):
    """
    Hierarchical 3D Gaussian Map containing Tier 0 (Global/Loop), Tier 1 (Structural), and Tier 2 (Detail).
    Includes a learned prior network for initial structural classification and residual rendering.
    """
    def __init__(self, max_tier1: int = 2000, max_tier2: int = 10000, max_tier0: int = 500, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.tier0 = GaussianTier(capacity=max_tier0, device=device) # Loop closure / Global
        self.tier1 = GaussianTier(capacity=max_tier1, device=device) # Coarse / Structural
        self.tier2 = GaussianTier(capacity=max_tier2, device=device) # Fine / Detail

        # Frozen learned prior for initializing structural vs detail assignment
        self.learned_prior = nn.Sequential(
            nn.Linear(3 + 1, 16), # Scale magnitude + Opacity -> Structural Score
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(device)
        self.learned_prior.eval() # Frozen prior

    def add_gaussians(self, xyz: torch.Tensor, scales: torch.Tensor, quats: torch.Tensor, 
                      opacities: torch.Tensor, colors: torch.Tensor, initial_tier: int = 2):
        """
        Adds new Gaussians into the map using the learned prior or specified tier.
        """
        N = xyz.shape[0]
        if N == 0:
            return

        target_tier = self.tier1 if initial_tier == 1 else self.tier2
        
        # Find inactive slots
        inactive_indices = torch.where(~target_tier.active_mask)[0]
        if len(inactive_indices) < N:
            N = len(inactive_indices)
            xyz, scales, quats, opacities, colors = xyz[:N], scales[:N], quats[:N], opacities[:N], colors[:N]
            
        slot_idx = inactive_indices[:N]

        with torch.no_grad():
            target_tier.xyz[slot_idx] = xyz
            target_tier.scales[slot_idx] = scales
            target_tier.quats[slot_idx] = quats
            target_tier.opacities[slot_idx] = opacities
            target_tier.colors[slot_idx] = colors
            target_tier.active_mask[slot_idx] = True
            target_tier.fisher_scores[slot_idx] = 0.0

    def predict_tier1_prior(self, scales: torch.Tensor, opacities: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the frozen learned prior: larger scale & high opacity -> higher Tier 1 structural probability.
        """
        with torch.no_grad():
            scale_exp = torch.exp(scales) # (N, 3)
            opac_sig = torch.sigmoid(opacities) # (N, 1)
            inputs = torch.cat([scale_exp, opac_sig], dim=-1) # (N, 4)
            prob_structural = self.learned_prior(inputs).squeeze(-1)
        return prob_structural

    def render_residual(
        self, 
        rasterizer: PyTorchGaussianRasterizer, 
        pose_cw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes NICE-SLAM residual composite rendering:
        I_total = I_Tier1 + (1 - Alpha_Tier1) * I_Tier2
        
        Returns:
            final_image: (3, H, W)
            img_tier1: (3, H, W)
            alpha_tier1: (1, H, W)
        """
        t1_dict = self.tier1.get_active_dict()
        t2_dict = self.tier2.get_active_dict()

        # Render Tier 1 Structural Backbone
        if self.tier1.num_active > 0:
            img_t1, alpha_t1 = rasterizer(
                t1_dict["xyz"], t1_dict["scales"], t1_dict["quats"],
                t1_dict["opacities"], t1_dict["colors"], pose_cw
            )
        else:
            H, W = rasterizer.H, rasterizer.W
            img_t1 = torch.zeros((3, H, W), device=self.device)
            alpha_t1 = torch.zeros((1, H, W), device=self.device)

        # Render Tier 2 Fine Details
        if self.tier2.num_active > 0:
            img_t2, alpha_t2 = rasterizer(
                t2_dict["xyz"], t2_dict["scales"], t2_dict["quats"],
                t2_dict["opacities"], t2_dict["colors"], pose_cw
            )
        else:
            H, W = rasterizer.H, rasterizer.W
            img_t2 = torch.zeros((3, H, W), device=self.device)

        # Residual Compositing
        final_image = img_t1 + (1.0 - alpha_t1) * img_t2
        return final_image, img_t1, alpha_t1
