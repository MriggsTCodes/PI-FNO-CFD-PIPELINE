"""
Loss Functions for the PI-FNO-CFD Pipeline
==========================================
Combined loss functions for training:
  1. Super-Resolution FNO (data + physics + boundary)
  2. PINN Refiner (residual correction + physics + boundary)
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from physics.navier_stokes import physics_loss
from physics.boundary import boundary_loss, wall_velocity_loss, cyclic_bc_loss


def sr_training_loss(sr_pred: torch.Tensor, fine_target: torch.Tensor,
                     fine_input: torch.Tensor,
                     mu_targets: torch.Tensor, sigma_targets: torch.Tensor,
                     physics_weight: float = 0.05,
                     boundary_weight: float = 5.0,
                     use_physics: bool = True) -> dict:
    """
    Combined loss for training the super-resolution FNO.
    
    Args:
        sr_pred:       (B, fine_nx, fine_ny, 4) — SR prediction (normalized)
        fine_target:   (B, fine_nx, fine_ny, 4) — Ground truth (normalized)
        fine_input:    (B, fine_nx, fine_ny, 5) — Fine grid coordinates
        mu_targets:    (4,) — Target channel means
        sigma_targets: (4,) — Target channel stds
        physics_weight: Weight for PDE residual loss
        boundary_weight: Weight for wall BC loss
        use_physics: Whether to include physics loss
        
    Returns:
        dict with 'total', 'data', 'physics', 'boundary' losses
    """
    # Data loss (supervised, in normalized space)
    loss_data = F.mse_loss(sr_pred, fine_target)
    
    # Physics loss (in physical units)
    if use_physics:
        preds_phys = sr_pred * sigma_targets + mu_targets
        loss_phys = physics_loss(preds_phys, fine_input)
    else:
        loss_phys = torch.tensor(0.0, device=sr_pred.device)
    
    # Boundary loss (wall BCs)
    loss_boundary = boundary_loss(sr_pred, fine_target, fine_input)
    
    # Total
    total = loss_data + physics_weight * loss_phys + boundary_weight * loss_boundary
    
    return {
        'total': total,
        'data': loss_data,
        'physics': loss_phys,
        'boundary': loss_boundary,
    }


def refiner_training_loss(refined_pred: torch.Tensor, sr_pred: torch.Tensor,
                          fine_target: torch.Tensor, fine_input: torch.Tensor,
                          mu_targets: torch.Tensor, sigma_targets: torch.Tensor,
                          physics_weight: float = 0.02,
                          boundary_weight: float = 0.05,
                          cyclic_weight: float = 1.0,
                          data_weight: float = 1.0) -> dict:
    """
    Combined loss for training the PINN refiner.
    
    The refiner is trained to:
      1. Match the ground truth (data loss)
      2. Minimize PDE residuals (physics loss)
      3. Enforce wall BCs (boundary loss)
      4. Strictly enforce cyclic periodicity (inlet-outlet matching)
      5. Improve upon the FNO baseline (correction should reduce error)
    """
    # Data loss
    loss_data = F.mse_loss(refined_pred, fine_target)
    
    # Physics loss (in physical units)
    preds_phys = refined_pred * sigma_targets + mu_targets
    loss_phys = physics_loss(preds_phys, fine_input)
    
    # Boundary loss
    loss_boundary = boundary_loss(refined_pred, fine_target, fine_input)
    
    # Wall velocity enforcement (no-slip)
    loss_wall_vel = wall_velocity_loss(
        preds_phys, fine_input
    )
    
    # Cyclic boundary loss (strict inlet/outlet periodicity on normalized fields)
    loss_cyclic = cyclic_bc_loss(refined_pred)
    
    # Total
    total = (data_weight * loss_data +
             physics_weight * loss_phys +
             boundary_weight * (loss_boundary + loss_wall_vel) +
             cyclic_weight * loss_cyclic)
    
    # Diagnostic: did the refiner improve over the FNO baseline?
    sr_error = F.mse_loss(sr_pred, fine_target).detach()
    ref_error = loss_data.detach()
    improvement = (sr_error - ref_error) / (sr_error + 1e-8)
    
    return {
        'total': total,
        'data': loss_data,
        'physics': loss_phys,
        'boundary': loss_boundary,
        'wall_velocity': loss_wall_vel,
        'cyclic': loss_cyclic,
        'improvement': improvement,
    }
