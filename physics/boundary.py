"""
Boundary Condition Enforcement
==============================
Functions for computing boundary condition losses at walls
and cyclic (periodic) boundaries of the channel.
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import T_WALL, SDF_BLEND_SHARPNESS


def wall_velocity_loss(preds: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Enforce no-slip condition at walls: u = v = 0.
    
    Uses the SDF (channel 4 of inputs) to identify wall-adjacent cells.
    The loss penalizes any non-zero velocity at points where SDF < 0.05.
    Scaled by 1000.0 so physical velocities (O(10^-3) m/s) match normalized loss scale.
    
    Args:
        preds:  (batch, nx, ny, 4) — [u, v, p, T] in physical units
        inputs: (batch, nx, ny, 5) — [x, y, T_wall, U_inlet, SDF]
        
    Returns:
        Scalar loss
    """
    sdf = inputs[..., 4]
    threshold = 0.05  # Normalized SDF threshold for "near wall"
    
    wall_mask = (sdf < threshold).float()
    n_wall = wall_mask.sum().clamp(min=1.0)
    
    u = preds[..., 0]
    v = preds[..., 1]
    
    # Physical velocity squared penalized near walls, scaled to O(10^-2)
    loss = 1000.0 * torch.sum(wall_mask * (u ** 2 + v ** 2)) / n_wall
    return loss


def wall_temperature_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Enforce wall temperature boundary condition.
    
    At the top and bottom grid boundaries (y=0 and y=H_max), the
    temperature should match the wall temperature.
    
    Args:
        preds:   (batch, nx, ny, 4) — Predicted fields
        targets: (batch, nx, ny, 4) — Ground truth fields
        
    Returns:
        Scalar loss
    """
    # Temperature at wall boundaries (first and last y-indices)
    loss = F.mse_loss(preds[:, :, 0, 3], targets[:, :, 0, 3]) + \
           F.mse_loss(preds[:, :, -1, 3], targets[:, :, -1, 3])
    return loss


def boundary_loss(preds: torch.Tensor, targets: torch.Tensor,
                  inputs: torch.Tensor = None) -> torch.Tensor:
    """
    Combined boundary loss at wall regions.
    
    If `inputs` is provided, uses SDF to identify wall-adjacent cells
    (correct for wavy channels where walls are NOT at fixed y-indices).
    Falls back to top/bottom grid indices if inputs not given.
    
    Args:
        preds:   (batch, nx, ny, 4)
        targets: (batch, nx, ny, 4)
        inputs:  (batch, nx, ny, 5) — optional, [x, y, T_wall, U_inlet, SDF]
        
    Returns:
        Scalar loss
    """
    if inputs is not None:
        sdf = inputs[..., 4]
        threshold = 0.05  # Normalized SDF threshold for "near wall"
        wall_mask = (sdf < threshold).unsqueeze(-1).float()  # (B, nx, ny, 1)
        n_wall = wall_mask.sum().clamp(min=1.0)
        n_channels = preds.shape[-1]
        loss = torch.sum(wall_mask * (preds - targets) ** 2) / (n_wall * n_channels)
    else:
        # Fallback: use top/bottom grid boundaries
        loss = F.mse_loss(preds[:, :, 0, :], targets[:, :, 0, :]) + \
               F.mse_loss(preds[:, :, -1, :], targets[:, :, -1, :])
    return loss


def cyclic_bc_loss(preds: torch.Tensor) -> torch.Tensor:
    """
    Enforce cyclic (periodic) boundary conditions.
    
    For a cyclic channel, fields at x=0 should match fields at x=L.
    This ensures the inlet and outlet of the periodic channel are consistent.
    
    Args:
        preds: (batch, nx, ny, 4) — [u, v, p, T]
        
    Returns:
        Scalar loss
    """
    loss = F.mse_loss(preds[:, 0, :, :], preds[:, -1, :, :])
    return loss
