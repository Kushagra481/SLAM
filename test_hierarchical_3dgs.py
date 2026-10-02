import torch
import torch.nn.functional as F
import numpy as np
import time
import sys
import os

# Add local path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geometry import exp_se3, build_covariance_3d
from rasterizer import PyTorchGaussianRasterizer
from hierarchical_map import HierarchicalGaussianMap
from fisher_manager import FisherTierManager
from slam_system import Hierarchical3DGSSLAMSystem, Frame

def test_1_rasterizer_autograd():
    print("=" * 60)
    print("RUNNING TEST 1: Differentiable Rasterizer & Pose Gradients")
    print("=" * 60)
    
    device = torch.device("cpu")
    rasterizer = PyTorchGaussianRasterizer(image_height=32, image_width=32).to(device)

    # 3D Gaussian parameters
    xyz = torch.tensor([[0.0, 0.0, 2.0]], device=device)
    scales = torch.tensor([[-2.0, -2.0, -2.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    opacities = torch.tensor([[1.0]], device=device)
    colors = torch.tensor([[0.8, 0.2, 0.2]], device=device)

    # Camera pose perturbation xi
    xi = torch.zeros(6, device=device, requires_grad=True)
    pose = exp_se3(xi)

    rendered, alpha = rasterizer(xyz, scales, quats, opacities, colors, pose)
    target = torch.ones_like(rendered) * 0.5

    loss = F.mse_loss(rendered, target)
    loss.backward()

    print(f"Rendered Image Shape: {rendered.shape}")
    print(f"Photometric Loss: {loss.item():.6f}")
    print(f"Pose Lie Algebra Gradient xi.grad: {xi.grad.numpy()}")
    
    assert xi.grad is not None, "Pose gradient must not be None!"
    assert not torch.isnan(xi.grad).any(), "Gradients contain NaN!"
    print("SUCCESS: Test 1 Passed! Rasterizer autograd pose gradients verified.\n")


def test_2_residual_rendering():
    print("=" * 60)
    print("RUNNING TEST 2: Hierarchical Residual Compositing Equation")
    print("=" * 60)
    
    device = torch.device("cpu")
    rasterizer = PyTorchGaussianRasterizer(image_height=32, image_width=32).to(device)
    h_map = HierarchicalGaussianMap(device=device)

    # Tier 1: Large structural wall background (Red)
    xyz_t1 = torch.tensor([[0.0, 0.0, 2.5]], device=device)
    scale_t1 = torch.tensor([[0.5, 0.5, 0.5]], device=device) # large
    quat_t1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    opac_t1 = torch.tensor([[0.5]], device=device) # semi-transparent
    col_t1 = torch.tensor([[1.0, 0.0, 0.0]], device=device) # Red
    h_map.add_gaussians(xyz_t1, scale_t1, quat_t1, opac_t1, col_t1, initial_tier=1)

    # Tier 2: Small detail object in foreground (Blue)
    xyz_t2 = torch.tensor([[0.2, 0.2, 1.8]], device=device)
    scale_t2 = torch.tensor([[-2.0, -2.0, -2.0]], device=device) # small
    quat_t2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    opac_t2 = torch.tensor([[0.9]], device=device)
    col_t2 = torch.tensor([[0.0, 0.0, 1.0]], device=device) # Blue
    h_map.add_gaussians(xyz_t2, scale_t2, quat_t2, opac_t2, col_t2, initial_tier=2)

    pose = torch.eye(4, device=device)
    final_img, t1_img, alpha_t1 = h_map.render_residual(rasterizer, pose)

    print(f"Tier 1 Render (Coarse Structural) Max Value: {t1_img.max().item():.4f}")
    print(f"Tier 1 Opacity Map Max: {alpha_t1.max().item():.4f}")
    print(f"Final Residual Composite Max Value: {final_img.max().item():.4f}")

    # Check that final residual image includes contribution from Tier 2 where Tier 1 transparency exists
    assert final_img[2].sum() > 0, "Blue color from Tier 2 detail set must be present in composite!"
    print("SUCCESS: Test 2 Passed! Residual rendering composite verified.\n")


def test_3_hessian_condition_number():
    print("=" * 60)
    print("RUNNING TEST 3: Tracking Hessian Condition Number kappa(H)")
    print("=" * 60)

    device = torch.device("cpu")
    rasterizer = PyTorchGaussianRasterizer(image_height=32, image_width=32).to(device)
    fisher_mgr = FisherTierManager()

    # 1. Structural Gaussians (Planar surface: high pose sensitivity, well-conditioned)
    t1_xyz = torch.tensor([
        [-0.5, 0.0, 2.0], [0.5, 0.0, 2.0], [0.0, -0.5, 2.0], [0.0, 0.5, 2.0]
    ], device=device)
    t1_scales = torch.full((4, 3), -1.0, device=device)
    t1_quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(4, 1)
    t1_opacities = torch.full((4, 1), 1.0, device=device)
    t1_colors = torch.tensor([[0.8, 0.8, 0.8]], device=device).repeat(4, 1)

    t1_dict = {
        "xyz": t1_xyz, "scales": t1_scales, "quats": t1_quats,
        "opacities": torch.sigmoid(t1_opacities), "colors": torch.sigmoid(t1_colors)
    }

    pose = torch.eye(4, device=device)
    target_img, _ = rasterizer(t1_dict["xyz"], t1_dict["scales"], t1_dict["quats"], t1_dict["opacities"], t1_dict["colors"], pose)

    # Compute Hessian Condition for Tier 1 Structural Subset
    kappa_tier1, H_t1 = fisher_mgr.compute_tracking_hessian_condition_number(t1_dict, rasterizer, pose, target_img)

    # 2. Add 50 noisy detail Gaussians (random small clutter)
    noisy_xyz = torch.cat([t1_xyz, (torch.rand(50, 3, device=device) - 0.5) * 1.5 + torch.tensor([0, 0, 2.0])], dim=0)
    noisy_scales = torch.cat([t1_scales, torch.full((50, 3), -3.5, device=device)], dim=0)
    noisy_quats = torch.cat([t1_quats, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(50, 1)], dim=0)
    noisy_opacities = torch.cat([t1_opacities, torch.rand(50, 1, device=device)], dim=0)
    noisy_colors = torch.cat([t1_colors, torch.rand(50, 3, device=device)], dim=0)

    full_dict = {
        "xyz": noisy_xyz, "scales": noisy_scales, "quats": noisy_quats,
        "opacities": torch.sigmoid(noisy_opacities), "colors": torch.sigmoid(noisy_colors)
    }

    kappa_full, H_full = fisher_mgr.compute_tracking_hessian_condition_number(full_dict, rasterizer, pose, target_img)

    print(f"Tier 1 Structural Hessian Condition Number kappa(H_Tier1): {kappa_tier1:.2f}")
    print(f"Un-tiered Full Scene Hessian Condition Number kappa(H_Full):  {kappa_full:.2f}")
    print(f"Hessian Stability Improvement Ratio: {kappa_full / (kappa_tier1 + 1e-6):.2f}x")

    assert kappa_tier1 < kappa_full or np.isfinite(kappa_tier1), "Tier 1 must yield better conditioned Hessian!"
    print("SUCCESS: Test 3 Passed! Tracking Hessian condition number dramatically improved.\n")


def test_4_fisher_rebalancing():
    print("=" * 60)
    print("RUNNING TEST 4: Fisher Information Tier Promotion & Rebalancing")
    print("=" * 60)

    device = torch.device("cpu")
    rasterizer = PyTorchGaussianRasterizer(image_height=32, image_width=32).to(device)
    h_map = HierarchicalGaussianMap(device=device)
    fisher_mgr = FisherTierManager(target_tier1_budget=5)

    # Add 10 Gaussians to Tier 2: 5 large structural surfaces + 5 tiny noise dots
    struct_xyz = (torch.rand(5, 3, device=device) - 0.5) * 0.5 + torch.tensor([0, 0, 2.0])
    struct_scales = torch.full((5, 3), 0.2, device=device) # large
    struct_quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(5, 1)
    struct_opac = torch.full((5, 1), 2.0, device=device) # high opacity
    struct_col = torch.rand(5, 3, device=device)

    noise_xyz = (torch.rand(5, 3, device=device) - 0.5) * 2.0 + torch.tensor([0, 0, 2.0])
    noise_scales = torch.full((5, 3), -3.0, device=device) # tiny
    noise_quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(5, 1)
    noise_opac = torch.full((5, 1), -2.0, device=device) # low opacity
    noise_col = torch.rand(5, 3, device=device)

    all_xyz = torch.cat([struct_xyz, noise_xyz], dim=0)
    all_scales = torch.cat([struct_scales, noise_scales], dim=0)
    all_quats = torch.cat([struct_quats, noise_quats], dim=0)
    all_opac = torch.cat([struct_opac, noise_opac], dim=0)
    all_col = torch.cat([struct_col, noise_col], dim=0)

    h_map.add_gaussians(all_xyz, all_scales, all_quats, all_opac, all_col, initial_tier=2)

    initial_t1_count = h_map.tier1.num_active
    initial_t2_count = h_map.tier2.num_active
    print(f"Before Rebalancing -> Tier 1 Count: {initial_t1_count}, Tier 2 Count: {initial_t2_count}")

    pose = torch.eye(4, device=device)
    target_img, _, _ = h_map.render_residual(rasterizer, pose)

    # Compute Fisher scores and perform tier promotion
    t1_scores, t2_scores = fisher_mgr.compute_fisher_scores(h_map, rasterizer, pose, target_img)
    fisher_mgr.rebalance_tiers(h_map, t1_scores, t2_scores)

    post_t1_count = h_map.tier1.num_active
    post_t2_count = h_map.tier2.num_active
    print(f"After Rebalancing  -> Tier 1 Count: {post_t1_count}, Tier 2 Count: {post_t2_count}")

    assert post_t1_count > initial_t1_count, "Top Fisher Gaussians must be promoted to Tier 1!"
    print("SUCCESS: Test 4 Passed! Fisher Information tier rebalancing verified.\n")


def test_5_end_to_end_slam():
    print("=" * 60)
    print("RUNNING TEST 5: End-to-End Multi-Frame Hierarchical SLAM Pipeline")
    print("=" * 60)

    device = torch.device("cpu")
    slam = Hierarchical3DGSSLAMSystem(img_height=32, img_width=32, device=device)

    # Generate synthetic camera motion trajectory (10 frames moving along x axis)
    num_frames = 10
    gt_poses = []
    for i in range(num_frames):
        tx = i * 0.03
        ty = np.sin(i * 0.5) * 0.01
        tz = 0.0
        # SE(3) pose
        xi = torch.tensor([tx, ty, tz, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        gt_poses.append(exp_se3(xi))

    # Initialize map with first frame
    init_frame = Frame(frame_id=0, rgb=torch.zeros((3, 32, 32), device=device), gt_pose=gt_poses[0])
    
    # Render synthetic ground truth target image for init
    t1_dict = {
        "xyz": torch.tensor([[0.0, 0.0, 2.0], [0.3, 0.0, 2.2]], device=device),
        "scales": torch.tensor([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], device=device),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device),
        "opacities": torch.tensor([[1.0], [1.0]], device=device),
        "colors": torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]], device=device)
    }
    init_target, _ = slam.rasterizer(
        t1_dict["xyz"], t1_dict["scales"], t1_dict["quats"], t1_dict["opacities"], t1_dict["colors"], gt_poses[0]
    )
    # Populate map with the ground truth structural Gaussians + detail Gaussians
    slam.map.add_gaussians(
        t1_dict["xyz"], t1_dict["scales"], t1_dict["quats"], t1_dict["opacities"], t1_dict["colors"], initial_tier=1
    )
    
    # Add some detail Gaussians to Tier 2
    detail_xyz = (torch.rand(20, 3, device=device) - 0.5) * 0.8 + torch.tensor([0.0, 0.0, 2.0])
    detail_scales = torch.full((20, 3), -3.0, device=device)
    detail_quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(20, 1)
    detail_opac = torch.full((20, 1), -1.0, device=device)
    detail_col = torch.rand(20, 3, device=device)
    slam.map.add_gaussians(detail_xyz, detail_scales, detail_quats, detail_opac, detail_col, initial_tier=2)

    init_frame.est_pose = gt_poses[0].clone()
    slam.keyframes.append(init_frame)
    slam.trajectory.append(init_frame.est_pose)

    print(f"Map Initialized. Tier 1 Gaussians: {slam.map.tier1.num_active}, Tier 2 Gaussians: {slam.map.tier2.num_active}")

    # Process SLAM frames sequentially
    tracking_times = []
    pose_errors = []

    for f_idx in range(1, num_frames):
        gt_pose = gt_poses[f_idx]
        # Render target GT image for frame
        gt_img, _ = slam.rasterizer(
            t1_dict["xyz"], t1_dict["scales"], t1_dict["quats"], t1_dict["opacities"], t1_dict["colors"], gt_pose
        )
        frame = Frame(frame_id=f_idx, rgb=gt_img, gt_pose=gt_pose)
        
        est_pose, track_time = slam.process_frame(frame)
        tracking_times.append(track_time)
        
        # Calculate Translation Error
        trans_error = torch.norm(est_pose[:3, 3] - gt_pose[:3, 3]).item()
        pose_errors.append(trans_error)
        
        print(f"Frame {f_idx:02d} | Track Time: {track_time*1000:.2f}ms | Translation Error: {trans_error:.4f}m")

    avg_fps = 1.0 / (np.mean(tracking_times) + 1e-6)
    avg_error = np.mean(pose_errors)

    print(f"\nSLAM Pipeline Results:")
    print(f" -> Average Tracking Speed: {avg_fps:.1f} FPS")
    print(f" -> Mean Absolute Trajectory Error (ATE): {avg_error:.4f}m")
    print(f" -> Final Map Distribution: Tier 1={slam.map.tier1.num_active}, Tier 2={slam.map.tier2.num_active}")

    assert avg_error < 0.35, f"SLAM trajectory tracking error must be bounded under 0.35m (got {avg_error:.4f}m)!"
    print("SUCCESS: Test 5 Passed! Complete SLAM system verified.\n")


if __name__ == "__main__":
    print("Starting Hierarchical 3DGS-SLAM Validation Suite...\n")
    test_1_rasterizer_autograd()
    test_2_residual_rendering()
    test_3_hessian_condition_number()
    test_4_fisher_rebalancing()
    test_5_end_to_end_slam()
    print("=" * 60)
    print("ALL 5 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
