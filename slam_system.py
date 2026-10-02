import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Optional
from geometry import exp_se3
from rasterizer import PyTorchGaussianRasterizer
from hierarchical_map import HierarchicalGaussianMap
from fisher_manager import FisherTierManager

class Frame:
    """Represents a SLAM input frame."""
    def __init__(self, frame_id: int, rgb: torch.Tensor, gt_pose: Optional[torch.Tensor] = None):
        self.frame_id = frame_id
        self.rgb = rgb # (3, H, W)
        self.gt_pose = gt_pose # (4, 4) ground truth pose if available
        self.est_pose = torch.eye(4, device=rgb.device) if gt_pose is None else gt_pose.clone()

class TrackingEngine:
    """
    Tracking Engine that estimates camera pose SE(3) by optimizing against
    ONLY the Tier 1 Coarse/Structural Gaussian subset.
    """
    def __init__(self, rasterizer: PyTorchGaussianRasterizer, num_iters: int = 30, lr: float = 0.05):
        self.rasterizer = rasterizer
        self.num_iters = num_iters
        self.lr = lr

    def track_frame(self, frame: Frame, hierarchical_map: HierarchicalGaussianMap, init_pose: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Optimizes SE(3) pose relative to init_pose using Tier 1 structural subset.
        Returns optimized pose T_cw and tracking time.
        """
        start_time = time.time()
        
        # Extract Tier 1 Gaussians ONLY (detached for tracking isolated to pose)
        t1_dict = hierarchical_map.tier1.get_active_dict()
        
        if len(t1_dict["xyz"]) == 0:
            # Fallback if Tier 1 is empty: use all active Gaussians
            t1_dict = hierarchical_map.tier2.get_active_dict()

        t1_dict = {k: v.detach() for k, v in t1_dict.items()}

        # Lie algebra perturbation parameter to optimize
        xi = torch.zeros(6, device=frame.rgb.device, requires_grad=True)
        optimizer = torch.optim.Adam([xi], lr=self.lr)

        best_loss = float("inf")
        best_pose = init_pose.clone()

        for step in range(self.num_iters):
            optimizer.zero_grad()
            
            # Compute perturbed pose T_cw = exp(xi) @ init_pose
            T_delta = exp_se3(xi)
            current_pose = torch.mm(T_delta, init_pose)
            
            # Render using Tier 1
            rendered, _ = self.rasterizer(
                t1_dict["xyz"], t1_dict["scales"], t1_dict["quats"],
                t1_dict["opacities"], t1_dict["colors"], current_pose
            )
            
            loss = torch.mean(torch.abs(rendered - frame.rgb))
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                with torch.no_grad():
                    best_pose = torch.mm(exp_se3(xi), init_pose).detach()

        elapsed_time = time.time() - start_time
        return best_pose, elapsed_time


class MappingEngine:
    """
    Mapping Engine that optimizes residual details in Tier 2 and triggers Fisher tier rebalancing.
    """
    def __init__(self, rasterizer: PyTorchGaussianRasterizer, fisher_manager: FisherTierManager):
        self.rasterizer = rasterizer
        self.fisher_manager = fisher_manager

    def process_keyframe(
        self, 
        keyframe: Frame, 
        hierarchical_map: HierarchicalGaussianMap, 
        num_mapping_iters: int = 10
    ):
        """
        Optimizes Tier 2 details while keeping Tier 1 frozen, then rebalances tiers based on Fisher scores.
        """
        t2_mask = hierarchical_map.tier2.active_mask
        if t2_mask.sum() == 0:
            return

        # Enable gradients for Tier 2 parameters
        optimizer = torch.optim.Adam([
            hierarchical_map.tier2.xyz,
            hierarchical_map.tier2.scales,
            hierarchical_map.tier2.opacities,
            hierarchical_map.tier2.colors
        ], lr=1e-3)

        for step in range(num_mapping_iters):
            optimizer.zero_grad()
            rendered, _, _ = hierarchical_map.render_residual(self.rasterizer, keyframe.est_pose)
            loss = torch.mean(torch.abs(rendered - keyframe.rgb))
            loss.backward()
            
            # Apply gradient mask to update active Tier 2 Gaussians only
            with torch.no_grad():
                hierarchical_map.tier2.xyz.grad *= t2_mask.unsqueeze(-1)
                hierarchical_map.tier2.scales.grad *= t2_mask.unsqueeze(-1)
                hierarchical_map.tier2.opacities.grad *= t2_mask.unsqueeze(-1)
                hierarchical_map.tier2.colors.grad *= t2_mask.unsqueeze(-1)

            optimizer.step()

        # Rebalance Tiers using Fisher Information
        t1_scores, t2_scores = self.fisher_manager.compute_fisher_scores(
            hierarchical_map, self.rasterizer, keyframe.est_pose, keyframe.rgb
        )
        self.fisher_manager.rebalance_tiers(hierarchical_map, t1_scores, t2_scores)


class Hierarchical3DGSSLAMSystem:
    """
    End-to-End Hierarchical 3DGS-SLAM System integrating:
    - Tier 1 Structural Backbone Tracking
    - Tier 2 Residual Detail Mapping
    - Fisher Information Tier Rebalancing
    """
    def __init__(self, img_height: int = 64, img_width: int = 64, device: torch.device = torch.device("cpu")):
        self.device = device
        self.rasterizer = PyTorchGaussianRasterizer(image_height=img_height, image_width=img_width).to(device)
        self.map = HierarchicalGaussianMap(device=device)
        self.fisher_manager = FisherTierManager(target_tier1_budget=50)
        
        self.tracker = TrackingEngine(self.rasterizer, num_iters=15, lr=0.01)
        self.mapper = MappingEngine(self.rasterizer, self.fisher_manager)
        
        self.keyframes: List[Frame] = []
        self.trajectory: List[torch.Tensor] = []

    def initialize_map(self, init_frame: Frame, num_gaussians: int = 200):
        """Initializes the scene map with synthetic/depth Gaussians."""
        device = self.device
        
        # Generate initial random 3D points
        xyz = (torch.rand(num_gaussians, 3, device=device) - 0.5) * 2.0
        xyz[:, 2] += 2.0 # In front of camera
        scales = torch.full((num_gaussians, 3), -2.5, device=device) # log scale ~0.08
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_gaussians, 1)
        opacities = torch.full((num_gaussians, 1), 1.0, device=device) # sigmoid(1.0) ~ 0.73
        colors = torch.rand(num_gaussians, 3, device=device)

        # Predict learned prior to populate Tier 1 vs Tier 2 initially
        prob_struct = self.map.predict_tier1_prior(scales, opacities)
        is_tier1 = prob_struct > 0.5

        t1_idx = torch.where(is_tier1)[0]
        t2_idx = torch.where(~is_tier1)[0]

        if len(t1_idx) > 0:
            self.map.add_gaussians(xyz[t1_idx], scales[t1_idx], quats[t1_idx], opacities[t1_idx], colors[t1_idx], initial_tier=1)
        if len(t2_idx) > 0:
            self.map.add_gaussians(xyz[t2_idx], scales[t2_idx], quats[t2_idx], opacities[t2_idx], colors[t2_idx], initial_tier=2)

        init_frame.est_pose = init_frame.gt_pose.clone() if init_frame.gt_pose is not None else torch.eye(4, device=device)
        self.keyframes.append(init_frame)
        self.trajectory.append(init_frame.est_pose)

    def process_frame(self, frame: Frame) -> Tuple[torch.Tensor, float]:
        """
        Process an incoming camera frame:
        1. Track frame pose using Tier 1 Gaussians only.
        2. If keyframe, trigger residual detail mapping & Fisher tier rebalancing.
        """
        last_pose = self.trajectory[-1]
        
        # 1. High-speed tracking using Tier 1
        est_pose, track_time = self.tracker.track_frame(frame, self.map, last_pose)
        frame.est_pose = est_pose
        self.trajectory.append(est_pose)

        # 2. Keyframe Mapping (every 3 frames)
        if frame.frame_id % 3 == 0:
            self.keyframes.append(frame)
            self.mapper.process_keyframe(frame, self.map)

        return est_pose, track_time
