"""
PINN Physics Refiner — Multi-Scale U-Net Architecture
======================================================
Post-processor that takes the FNO super-resolution output and
applies physics-based corrections using PDE residual minimization.

Architecture v3: Multi-Scale Cyclic U-Net
  - Encoder path with downsampling (128×64 → 64×32 → 32×16) for global context
  - Bottleneck at 32×16 gives full-domain receptive field for elliptic pressure
  - Decoder path with skip connections for preserving fine-scale detail
  - All convolutions use cyclic x-padding and replicate y-padding
  - Zero-initialized output projection for identity-start training

This design directly addresses the elliptic pressure bottleneck:
the previous flat CyclicResBlock had a 9×9 receptive field, insufficient
for propagating global pressure information governed by the Poisson equation.
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
    if pad <= 0:
        return x
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


class CyclicDownBlock(nn.Module):
    """Encoder block: CyclicResBlock at input resolution, then downsample.
    
    Returns (downsampled_output, skip_connection).
    The skip connection is at the INPUT channel width (before projection),
    so the decoder knows exactly what channel count to expect.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.res = CyclicResBlock(in_ch)
        # Channel projection applied AFTER skip tap
        self.proj = CyclicConv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.norm = nn.GroupNorm(4, out_ch) if in_ch != out_ch else nn.Identity()
        # Downsample: average pool 2×2 (preserves cyclic structure)
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.skip_ch = in_ch  # expose for decoder construction
    
    def forward(self, x: torch.Tensor):
        h = self.res(x)
        skip = h                        # skip at in_ch channels
        h = self.norm(self.proj(h))     # project to out_ch
        return self.down(h), skip       # return (downsampled at out_ch, skip at in_ch)


class CyclicUpBlock(nn.Module):
    """Decoder block: upsample + concat skip + CyclicResBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up_proj = CyclicConv2d(in_ch, out_ch, kernel_size=1)
        self.up_norm = nn.GroupNorm(4, out_ch)
        self.res = CyclicResBlock(out_ch + skip_ch)
        # Project merged channels back to out_ch
        self.merge_proj = CyclicConv2d(out_ch + skip_ch, out_ch, kernel_size=1)
        self.merge_norm = nn.GroupNorm(4, out_ch)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        # Upsample via bilinear interpolation (resolution-independent)
        h = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        h = self.up_norm(self.up_proj(h))
        # Concatenate skip connection
        h = torch.cat([h, skip], dim=1)
        h = self.res(h)
        h = self.merge_norm(self.merge_proj(h))
        return h


class PINN_Refiner(nn.Module):
    """
    Multi-Scale Cyclic U-Net Physics Refiner.
    
    Addresses the elliptic pressure bottleneck by providing a global
    receptive field through encoder-decoder architecture with skip
    connections, while maintaining exact cyclic periodicity via
    CyclicConv2d padding.
    
    Encoder path: 128×64 → 64×32 → 32×16 (global context for pressure)
    Bottleneck:   32×16 with 2× CyclicResBlock (full-domain view)
    Decoder path: 32×16 → 64×32 → 128×64 (reconstruct fine corrections)
    
    Args:
        in_features: Number of input channels (5 coords + 4 fields = 9)
        hidden_features: Base channel width (scales 1×, 2×, 4× through encoder)
        out_features: Number of output correction channels (4)
        num_layers: Ignored for backward compat; architecture is fixed U-Net
    """
    
    def __init__(self, in_features: int = 9, hidden_features: int = 64,
                 out_features: int = 4, num_layers: int = 4):
        super().__init__()
        
        C = hidden_features  # Base channels: 64
        
        # ── Input projection ──
        self.in_proj = nn.Sequential(
            CyclicConv2d(in_features, C, kernel_size=3),
            nn.GroupNorm(4, C),
            nn.GELU()
        )
        
        # ── Encoder (downsampling) path ──
        self.enc1 = CyclicDownBlock(C, C)          # 128×64 → 64×32, skip @ C channels
        self.enc2 = CyclicDownBlock(C, 2 * C)      # 64×32 → 32×16,  skip @ C channels
        
        # ── Bottleneck at 32×16 (global receptive field for pressure) ──
        self.bottleneck = nn.Sequential(
            CyclicResBlock(2 * C),
            CyclicResBlock(2 * C),
        )
        
        # ── Decoder (upsampling) path ──
        # skip channels match enc's INPUT channels (before projection)
        self.dec2 = CyclicUpBlock(2 * C, C, C)     # 32×16 → 64×32  (skip2 has C ch)
        self.dec1 = CyclicUpBlock(C, C, C)          # 64×32 → 128×64 (skip1 has C ch)
        
        # ── Output projection (zero-initialized) ──
        self.out_conv = CyclicConv2d(C, out_features, kernel_size=3)
        nn.init.zeros_(self.out_conv.conv.weight)
        nn.init.zeros_(self.out_conv.conv.bias)
    
    def load_state_dict(self, state_dict, strict=False):
        """Robust state_dict loader that supports legacy checkpoints."""
        # Check if this is a legacy flat-network checkpoint
        is_legacy_flat = any(k.startswith('net.') for k in state_dict.keys())
        is_legacy_res = (any(k.startswith('in_proj.') for k in state_dict.keys()) and
                         not any(k.startswith('enc1.') for k in state_dict.keys()))
        
        if is_legacy_flat or is_legacy_res:
            # Legacy checkpoint incompatible with U-Net; start fresh
            print("[PINN_Refiner] Legacy checkpoint detected — starting with fresh U-Net weights (zero-init output).")
            return  # Skip loading; use freshly initialized weights
        
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
        
        # Input projection
        h = self.in_proj(x)
        
        # Encoder
        h1, skip1 = self.enc1(h)       # skip1 @ 128×64, h1 @ 64×32
        h2, skip2 = self.enc2(h1)      # skip2 @ 64×32,  h2 @ 32×16
        
        # Bottleneck (global context)
        h2 = self.bottleneck(h2)
        
        # Decoder
        d2 = self.dec2(h2, skip2)      # 32×16 → 64×32
        d1 = self.dec1(d2, skip1)      # 64×32 → 128×64
        
        # Output correction
        correction = self.out_conv(d1).permute(0, 2, 3, 1)  # (B, nx, ny, 4)
        
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
        h1, skip1 = self.enc1(h)
        h2, skip2 = self.enc2(h1)
        h2 = self.bottleneck(h2)
        d2 = self.dec2(h2, skip2)
        d1 = self.dec1(d2, skip1)
        return self.out_conv(d1).permute(0, 2, 3, 1)


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
