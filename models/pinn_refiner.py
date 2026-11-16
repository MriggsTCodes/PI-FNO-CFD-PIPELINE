"""
PINN Physics Refiner
====================
Post-processor that takes the FNO super-resolution output and
applies physics-based corrections using PDE residual minimization.

Two modes of operation:
  1. Learned Refiner (training): MLP predicts a correction field that
     is added to the FNO output. Trained with physics + data loss.
  2. Iterative Refiner (inference): Gradient-based optimization of the
     solution to minimize PDE residuals (no learned parameters needed).

The Learned Refiner uses SDF-weighted blending: stronger corrections
near walls (where FNO struggles) and minimal corrections in the interior
(where FNO is already accurate).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SDF_BLEND_SHARPNESS, N_INPUT_CHANNELS, N_OUTPUT_CHANNELS


def pad_channel(x: torch.Tensor, pad: int = 1) -> torch.Tensor:
    """Circular padding along x (dim 2), replicate padding along y (dim 3)."""
    x = torch.cat([x[:, :, -pad:, :], x, x[:, :, :pad, :]], dim=2)
    x = F.pad(x, [pad, pad, 0, 0], mode='replicate')
    return x


class CyclicConv2d(nn.Module):
    """2D Convolution with cyclic boundary condition along x, replicate along y."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(pad_channel(x, self.pad))


class CyclicResBlock(nn.Module):
    """Residual block with cyclic boundary condition along x and GroupNorm."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = CyclicConv2d(channels, channels, kernel_size=3)
        self.norm1 = nn.GroupNorm(4, channels)
        self.act1 = nn.GELU()
        self.conv2 = CyclicConv2d(channels, channels, kernel_size=3)
        self.norm2 = nn.GroupNorm(4, channels)
        self.act2 = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + res)


class PINN_Refiner(nn.Module):
    """
    Learned physics-informed correction network.
    
    Takes the FNO super-resolution output + grid metadata as input,
    and outputs a residual correction Δ(u, v, p, T) that is added
    to the FNO prediction.
    
    Uses 2D cyclic residual convolutions to preserve spatial gradients and
    channel periodicity, avoiding checkerboard artifacts of pointwise MLPs.
    
    Args:
        in_features: Number of input channels (5 coords + 4 fields = 9)
        hidden_features: Channel width of convolutional layers
        out_features: Number of output correction channels (4)
        num_layers: Depth of convolutional network
    """
    
    def __init__(self, in_features: int = 9, hidden_features: int = 64,
                 out_features: int = 4, num_layers: int = 4):
        super().__init__()
        
        self.in_proj = nn.Sequential(
            CyclicConv2d(in_features, hidden_features, kernel_size=3),
            nn.GroupNorm(4, hidden_features),
            nn.GELU()
        )
        
        # Stack of residual blocks
        n_blocks = max(1, (num_layers - 2) // 2)
        blocks = [CyclicResBlock(hidden_features) for _ in range(n_blocks)]
        self.res_blocks = nn.Sequential(*blocks)
        
        self.out_conv = CyclicConv2d(hidden_features, out_features, kernel_size=3)
        # Initialize final layer to zero (start with exactly zero correction)
        nn.init.zeros_(self.out_conv.conv.weight)
        nn.init.zeros_(self.out_conv.conv.bias)
    
    def load_state_dict(self, state_dict, strict=False):
        """Robust state_dict loader that supports both legacy sequential and residual checkpoints."""
        is_legacy = any(k.startswith('net.') for k in state_dict.keys())
        is_current_res = hasattr(self, 'in_proj')
        
        if is_legacy and is_current_res:
            mapped_dict = {}
            for k, v in state_dict.items():
                new_k = k
                new_k = new_k.replace('net.0.', 'in_proj.0.')
                new_k = new_k.replace('net.2.', 'res_blocks.0.conv1.')
                new_k = new_k.replace('net.3.', 'res_blocks.0.norm1.')
                new_k = new_k.replace('net.5.', 'res_blocks.0.conv2.')
                new_k = new_k.replace('net.6.', 'res_blocks.0.norm2.')
                new_k = new_k.replace('net.8.', 'out_conv.')
                mapped_dict[new_k] = v
            return super().load_state_dict(mapped_dict, strict=False)
        return super().load_state_dict(state_dict, strict=strict)
    
    def forward(self, fno_pred: torch.Tensor, grid_input: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-corrected prediction.
        
        Args:
            fno_pred:   (batch, nx, ny, 4) — FNO super-resolution output
            grid_input: (batch, nx, ny, 5) — [x, y, T_wall, U_inlet, SDF]
            
        Returns:
            refined:    (batch, nx, ny, 4) — Physics-corrected fields
        """
        # Concatenate FNO prediction + grid metadata: (B, nx, ny, 9)
        refiner_input = torch.cat([grid_input, fno_pred], dim=-1)
        
        # Permute to (B, C, nx, ny) for 2D convolution
        x = refiner_input.permute(0, 3, 1, 2)
        h = self.in_proj(x)
        h = self.res_blocks(h)
        correction = self.out_conv(h).permute(0, 2, 3, 1)  # (B, nx, ny, 4)
        
        # SDF-weighted blending
        sdf = grid_input[..., 4:5]  # (B, nx, ny, 1)
        alpha = torch.sigmoid(5.0 * (sdf - 0.1))
        wall_weight = (1.0 - alpha) * 0.8 + 0.2
        
        refined = fno_pred + wall_weight * correction
        return refined
    
    def get_correction(self, fno_pred: torch.Tensor, grid_input: torch.Tensor) -> torch.Tensor:
        """Return only the correction field (for diagnostics)."""
        refiner_input = torch.cat([grid_input, fno_pred], dim=-1)
        x = refiner_input.permute(0, 3, 1, 2)
        h = self.in_proj(x)
        h = self.res_blocks(h)
        return self.out_conv(h).permute(0, 2, 3, 1)


class IterativeRefiner:
    """
    Gradient-based iterative physics refinement (no learned parameters).
    
    Starting from the FNO prediction, performs gradient descent on the
    PDE residuals to push the solution towards physical consistency.
    
    This is a test-time optimization — slower but requires no training.
    
    Args:
        physics_loss_fn: Function(preds, inputs) → scalar PDE residual loss
        n_steps: Number of refinement iterations
        lr: Step size for gradient updates
        boundary_weight: Weight for boundary condition enforcement
    """
    
    def __init__(self, physics_loss_fn, n_steps: int = 100,
                 lr: float = 1e-3, boundary_weight: float = 10.0):
        self.physics_loss_fn = physics_loss_fn
        self.n_steps = n_steps
        self.lr = lr
        self.boundary_weight = boundary_weight
    
    def refine(self, fno_pred: torch.Tensor, inputs: torch.Tensor,
               mu_targets: torch.Tensor, sigma_targets: torch.Tensor) -> torch.Tensor:
        """
        Iteratively refine the FNO prediction by minimizing PDE residuals.
        
        Args:
            fno_pred:      (batch, nx, ny, 4) — Initial FNO prediction (normalized)
            inputs:        (batch, nx, ny, 5) — Grid coordinates + BCs
            mu_targets:    (4,) — Target means for de-normalization
            sigma_targets: (4,) — Target stds for de-normalization
            
        Returns:
            refined: (batch, nx, ny, 4) — Refined prediction (normalized)
        """
        # Make a copy that requires grad for optimization
        refined = fno_pred.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([refined], lr=self.lr)
        
        for step in range(self.n_steps):
            optimizer.zero_grad()
            
            # De-normalize for physics computation
            preds_phys = refined * sigma_targets + mu_targets
            
            # PDE residual loss
            loss_phys = self.physics_loss_fn(preds_phys, inputs)
            
            # Boundary condition loss (wall velocities should be zero)
            loss_bc = F.mse_loss(refined[:, :, 0, :], fno_pred[:, :, 0, :]) + \
                      F.mse_loss(refined[:, :, -1, :], fno_pred[:, :, -1, :])
            
            loss = loss_phys + self.boundary_weight * loss_bc
            loss.backward()
            optimizer.step()
        
        return refined.detach()
