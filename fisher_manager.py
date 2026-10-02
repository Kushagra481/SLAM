import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from geometry import exp_se3
from rasterizer import PyTorchGaussianRasterizer
from hierarchical_map import HierarchicalGaussianMap

class FisherTierManager:
    """
    Manages Fisher-information-driven Gaussian tier promotion/demotion
    and Tracking Hessian condition analysis.
    """
    def __init__(self, target_tier1_budget: int = 150):
        self.target_tier1_budget = target_tier1_budget

    def compute_fisher_scores(
        self, 
        hierarchical_map: HierarchicalGaussianMap, 
        rasterizer: PyTorchGaussianRasterizer, 
        pose_cw: torch.Tensor, 
        target_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Fisher Information scores for active Gaussians in Tier 1 and Tier 2.
        
        Fisher score F_i measures sensitivity of photometric loss w.r.t pose & position:
        F_i = || d(L_photo) / d(xyz_i) ||_2^2
        """
        # Create zero perturbation for Lie algebra tracking pose sensitivity
        xi = torch.zeros(6, device=hierarchical_map.device, requires_grad=True)
        T_delta = exp_se3(xi)
        current_pose_perturbed = torch.mm(T_delta, pose_cw)

        # Clone positions to enable gradient tracking
        t1_dict = hierarchical_map.tier1.get_active_dict()
        t2_dict = hierarchical_map.tier2.get_active_dict()

        if hierarchical_map.tier2.num_active == 0 and hierarchical_map.tier1.num_active == 0:
            return torch.tensor([]), torch.tensor([])

        # Render combined scene with perturbation
        rendered_image, _, _ = hierarchical_map.render_residual(rasterizer, current_pose_perturbed)
        
        # Photometric loss
        loss = F.l1_loss(rendered_image, target_img)
        
        # Compute gradient w.r.t. pose perturbation xi
        grad_xi = torch.autograd.grad(loss, xi, create_graph=True)[0]
        
        # Compute sensitivity score per Gaussian in Tier 2
        t2_mask = hierarchical_map.tier2.active_mask
        t2_xyz = hierarchical_map.tier2.xyz[t2_mask]
        
        if len(t2_xyz) > 0 and t2_xyz.requires_grad:
            # Fisher proxy: norm of position gradient weighted by pose sensitivity
            grad_t2_xyz = torch.autograd.grad(grad_xi.pow(2).sum(), t2_xyz, retain_graph=True, allow_unused=True)[0]
            if grad_t2_xyz is not None:
                t2_scores = grad_t2_xyz.pow(2).sum(dim=-1).detach()
            else:
                t2_scores = torch.zeros(len(t2_xyz), device=hierarchical_map.device)
        else:
            t2_scores = torch.zeros(hierarchical_map.tier2.num_active, device=hierarchical_map.device)

        # Compute sensitivity score per Gaussian in Tier 1
        t1_mask = hierarchical_map.tier1.active_mask
        t1_xyz = hierarchical_map.tier1.xyz[t1_mask]
        if len(t1_xyz) > 0 and t1_xyz.requires_grad:
            grad_t1_xyz = torch.autograd.grad(grad_xi.pow(2).sum(), t1_xyz, retain_graph=False, allow_unused=True)[0]
            if grad_t1_xyz is not None:
                t1_scores = grad_t1_xyz.pow(2).sum(dim=-1).detach()
            else:
                t1_scores = torch.ones(len(t1_xyz), device=hierarchical_map.device)
        else:
            t1_scores = torch.ones(hierarchical_map.tier1.num_active, device=hierarchical_map.device)

        return t1_scores, t2_scores

    def compute_tracking_hessian_condition_number(
        self, 
        gaussians_dict: Dict[str, torch.Tensor], 
        rasterizer: PyTorchGaussianRasterizer, 
        pose_cw: torch.Tensor, 
        target_img: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        """
        Computes the 6x6 Pose Tracking Hessian Matrix H = J^T J and its Condition Number kappa(H).
        Hessian condition number kappa(H) = lambda_max / lambda_min.
        A smaller condition number indicates a far more stable pose optimization problem.
        """
        if len(gaussians_dict["xyz"]) == 0:
            return float("inf"), torch.eye(6)

        xi = torch.zeros(6, device=pose_cw.device, requires_grad=True)
        T_delta = exp_se3(xi)
        perturbed_pose = torch.mm(T_delta, pose_cw)

        rendered, _ = rasterizer(
            gaussians_dict["xyz"], gaussians_dict["scales"], gaussians_dict["quats"],
            gaussians_dict["opacities"], gaussians_dict["colors"], perturbed_pose
        )
        
        residuals = (rendered - target_img).reshape(-1) # (3 * H * W,)
        
        # Compute Jacobian J: (N_residuals, 6)
        J_list = []
        for r_idx in range(0, len(residuals), max(1, len(residuals) // 50)): # Subsample pixels for numerical fast evaluation
            r_val = residuals[r_idx]
            g = torch.autograd.grad(r_val, xi, retain_graph=True)[0]
            J_list.append(g)

        if len(J_list) == 0:
            return float("inf"), torch.eye(6)

        J = torch.stack(J_list, dim=0) # (M, 6)
        H = torch.mm(J.T, J) # (6, 6)

        # Eigenvalue analysis
        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(H)
            eigvals = torch.clamp(eigvals, min=1e-8)
            kappa = (eigvals.max() / eigvals.min()).item()

        return kappa, H.detach()

    def rebalance_tiers(
        self, 
        hierarchical_map: HierarchicalGaussianMap, 
        t1_scores: torch.Tensor, 
        t2_scores: torch.Tensor
    ):
        """
        Promotes high-Fisher Gaussians from Tier 2 to Tier 1, and demotes low-Fisher Gaussians from Tier 1 to Tier 2.
        Enforces target_tier1_budget.
        """
        t2_mask = torch.where(hierarchical_map.tier2.active_mask)[0]
        t1_mask = torch.where(hierarchical_map.tier1.active_mask)[0]

        if len(t2_mask) == 0:
            return

        # 1. Evaluate Learned Prior + Fisher Scores for candidate promotion
        t2_scales = hierarchical_map.tier2.scales[t2_mask]
        t2_opacities = hierarchical_map.tier2.opacities[t2_mask]
        prior_scores = hierarchical_map.predict_tier1_prior(t2_scales, t2_opacities)
        
        combined_t2_scores = prior_scores * 0.5 + (t2_scores / (t2_scores.max() + 1e-6)) * 0.5

        # 2. Select top K candidates to promote
        k_promote = min(self.target_tier1_budget - len(t1_mask), len(t2_mask))
        if k_promote > 0:
            top_k_val, top_k_local_idx = torch.topk(combined_t2_scores, k=k_promote)
            promote_global_idx = t2_mask[top_k_local_idx]

            # Copy parameters from Tier 2 to Tier 1
            t1_inactive = torch.where(~hierarchical_map.tier1.active_mask)[0][:k_promote]
            
            with torch.no_grad():
                hierarchical_map.tier1.xyz[t1_inactive] = hierarchical_map.tier2.xyz[promote_global_idx]
                hierarchical_map.tier1.scales[t1_inactive] = hierarchical_map.tier2.scales[promote_global_idx]
                hierarchical_map.tier1.quats[t1_inactive] = hierarchical_map.tier2.quats[promote_global_idx]
                hierarchical_map.tier1.opacities[t1_inactive] = hierarchical_map.tier2.opacities[promote_global_idx]
                hierarchical_map.tier1.colors[t1_inactive] = hierarchical_map.tier2.colors[promote_global_idx]
                hierarchical_map.tier1.active_mask[t1_inactive] = True
                
                # Deactivate in Tier 2
                hierarchical_map.tier2.active_mask[promote_global_idx] = False
