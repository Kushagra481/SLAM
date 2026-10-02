import torch
import torch.nn.functional as F

def exp_se3(xi: torch.Tensor) -> torch.Tensor:
    """
    Computes the SE(3) matrix from Lie algebra vector xi in se(3).
    xi: shape (6,) where xi[:3] is translation v, xi[3:] is rotation vector omega.
    Returns: T of shape (4, 4) in SE(3).
    """
    xi = xi.to(torch.float32)
    v = xi[:3]
    omega = xi[3:]
    theta = torch.norm(omega)

    T = torch.eye(4, device=xi.device, dtype=xi.dtype)

    if theta < 1e-6:
        T[:3, 3] = v
        return T

    axis = omega / theta
    wx = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], device=xi.device, dtype=xi.dtype)

    # Rodrigues formula for rotation matrix R
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    R = torch.eye(3, device=xi.device, dtype=xi.dtype) + sin_t * wx + (1 - cos_t) * torch.mm(wx, wx)

    # V matrix for translation
    V = torch.eye(3, device=xi.device, dtype=xi.dtype) + ((1 - cos_t) / theta) * wx + ((theta - sin_t) / theta) * torch.mm(wx, wx)
    t = torch.mv(V, v)

    T[:3, :3] = R
    T[:3, 3] = t
    return T

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    Converts normalized quaternion q (w, x, y, z) to 3x3 rotation matrix.
    q: shape (..., 4)
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.stack([
        1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))

    return R

def build_covariance_3d(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """
    Computes 3D Covariance Matrix Sigma = R S S^T R^T for Gaussians.
    scales: (N, 3)
    quats: (N, 4)
    Returns: Sigma of shape (N, 3, 3)
    """
    S = torch.diag_embed(torch.exp(scales)) # (N, 3, 3)
    R = quat_to_rotmat(quats)              # (N, 3, 3)
    M = torch.bmm(R, S)                     # (N, 3, 3)
    Sigma = torch.bmm(M, M.transpose(1, 2)) # (N, 3, 3)
    return Sigma
