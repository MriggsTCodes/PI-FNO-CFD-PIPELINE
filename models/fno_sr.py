"""
FNO Super-Resolution Model
===========================
Takes a coarse CFD solution (32×16) and upscales it to fine resolution
(128×64) using a Fourier Neural Operator with learned spectral upsampling.

Architecture:
  1. Lifting: Project coarse input channels to a high-dimensional feature space
  2. Spectral Blocks: 4 Fourier layers that learn global patterns at coarse resolution
  3. Spectral Upsampling: Bilinear interpolation to fine resolution in feature space
  4. Fine Spectral Blocks: 2 additional Fourier layers at fine resolution for detail
  5. Projection: Map features back to physical field outputs (u, v, p, T)

The key insight is that FNO's spectral conv layers are resolution-invariant —
they operate on Fourier modes, so the same weights work at any grid size.
We exploit this by processing at coarse resolution first (cheap), then
upsampling the features and refining at fine resolution (captures details).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spectral_conv import SpectralConv2d


class FNO_SuperResolution(nn.Module):
    """
    Fourier Neural Operator for CFD Super-Resolution.
    
    Coarse (32×16) input → Fine (128×64) output.
    
    The model takes coarse CFD fields + coordinates + BCs as input
    and outputs high-resolution field predictions.
    
    Input channels (9): [x, y, T_wall, U_inlet, SDF, u_coarse, v_coarse, p_coarse, T_coarse]
    Output channels (4): [u, v, p, T] at fine resolution
    
    Args:
        modes1, modes2: Number of Fourier modes
        width: Channel width in the spectral layers
        fine_nx, fine_ny: Target fine resolution
    """
    
    def __init__(self, modes1: int = 12, modes2: int = 12,
                 width: int = 48, fine_nx: int = 128, fine_ny: int = 64):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fine_nx = fine_nx
        self.fine_ny = fine_ny
        self.padding = 2
        
        # Input: coarse fields (9 channels) = 5 input + 4 coarse fields
        n_input = 9
        
        # ── Lifting Layer ──
        self.lift = nn.Linear(n_input, width)
        
        # ── Coarse-Resolution Spectral Blocks (process at 32×16) ──
        self.conv_c0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv_c1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv_c2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv_c3 = SpectralConv2d(width, width, modes1, modes2)
        
        self.mlp_c0 = nn.Conv2d(width, width, 1)
        self.mlp_c1 = nn.Conv2d(width, width, 1)
        self.mlp_c2 = nn.Conv2d(width, width, 1)
        self.mlp_c3 = nn.Conv2d(width, width, 1)
        
        # ── Fine-Resolution Spectral Blocks (refine at 128×64) ──
        self.conv_f0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv_f1 = SpectralConv2d(width, width, modes1, modes2)
        
        self.mlp_f0 = nn.Conv2d(width, width, 1)
        self.mlp_f1 = nn.Conv2d(width, width, 1)
        
        # ── Skip Connection: bilinear-upsampled coarse features ──
        self.skip_proj = nn.Conv2d(width, width, 1)
        
        # ── Projection Layer ──
        self.proj1 = nn.Linear(width, 128)
        self.proj2 = nn.Linear(128, 4)
    
    def forward(self, coarse_input: torch.Tensor, fine_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coarse_input: (batch, coarse_nx, coarse_ny, 9) 
                          [x, y, T_wall, U_inlet, SDF, u, v, p, T] on coarse grid
            fine_coords:  (batch, fine_nx, fine_ny, 5) 
                          [x, y, T_wall, U_inlet, SDF] on fine grid
                          
        Returns:
            fine_pred: (batch, fine_nx, fine_ny, 4) — [u, v, p, T] on fine grid
        """
        # ── 1. Lifting at coarse resolution ──
        x = self.lift(coarse_input)                    # (B, cx, cy, width)
        x = x.permute(0, 3, 1, 2)                     # (B, width, cx, cy)
        
        # Pad for non-periodic boundaries
        x = F.pad(x, [0, self.padding, 0, self.padding])
        
        # ── 2. Coarse Spectral Blocks ──
        x1 = self.conv_c0(x); x2 = self.mlp_c0(x); x = F.gelu(x1 + x2)
        x1 = self.conv_c1(x); x2 = self.mlp_c1(x); x = F.gelu(x1 + x2)
        x1 = self.conv_c2(x); x2 = self.mlp_c2(x); x = F.gelu(x1 + x2)
        x1 = self.conv_c3(x); x2 = self.mlp_c3(x); x = F.gelu(x1 + x2)
        
        # Un-pad
        x = x[..., :-self.padding, :-self.padding]
        
        # Save coarse features for skip connection
        coarse_features = x
        
        # ── 3. Spectral Upsampling (coarse → fine) ──
        # Bilinear interpolation in feature space
        x = F.interpolate(
            x, size=(self.fine_nx, self.fine_ny),
            mode='bilinear', align_corners=True
        )  # (B, width, fine_nx, fine_ny)
        
        # Skip connection: upsampled coarse features
        skip = F.interpolate(
            coarse_features, size=(self.fine_nx, self.fine_ny),
            mode='bilinear', align_corners=True
        )
        skip = self.skip_proj(skip)
        
        # Pad for fine spectral blocks
        x = F.pad(x, [0, self.padding, 0, self.padding])
        
        # ── 4. Fine Spectral Blocks ──
        x1 = self.conv_f0(x); x2 = self.mlp_f0(x); x = F.gelu(x1 + x2)
        x1 = self.conv_f1(x); x2 = self.mlp_f1(x); x = F.gelu(x1 + x2)
        
        # Un-pad
        x = x[..., :-self.padding, :-self.padding]
        
        # Add skip connection
        x = x + skip
        
        # ── 5. Projection ──
        x = x.permute(0, 2, 3, 1)                     # (B, fine_nx, fine_ny, width)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)                              # (B, fine_nx, fine_ny, 4)
        
        return x


class FNO_SuperResolution_Simple(nn.Module):
    """
    Simplified Super-Resolution FNO — processes entirely at fine resolution.
    
    Uses bilinear upsampling of the coarse input BEFORE the spectral blocks,
    then processes all FNO layers at the target fine resolution.
    Simpler but more memory-intensive than the hierarchical version.
    
    Input: Upsampled coarse solution concatenated with fine grid coordinates.
    Output: Corrected fine-resolution fields.
    """
    
    def __init__(self, modes1: int = 12, modes2: int = 12,
                 width: int = 48, fine_nx: int = 128, fine_ny: int = 64):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fine_nx = fine_nx
        self.fine_ny = fine_ny
        self.padding = 2
        
        # Input: fine coords (5) + upsampled coarse fields (4) = 9
        self.lift = nn.Linear(9, width)
        
        # Spectral blocks at fine resolution
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        
        self.mlp0 = nn.Conv2d(width, width, 1)
        self.mlp1 = nn.Conv2d(width, width, 1)
        self.mlp2 = nn.Conv2d(width, width, 1)
        self.mlp3 = nn.Conv2d(width, width, 1)
        
        self.proj1 = nn.Linear(width, 128)
        self.proj2 = nn.Linear(128, 4)
    
    def forward(self, coarse_input: torch.Tensor, fine_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coarse_input: (batch, coarse_nx, coarse_ny, 9)
            fine_coords:  (batch, fine_nx, fine_ny, 5)
            
        Returns:
            fine_pred: (batch, fine_nx, fine_ny, 4)
        """
        B = coarse_input.shape[0]
        
        # Upsample coarse fields to fine resolution
        coarse_fields = coarse_input[..., 5:9]  # (B, cx, cy, 4)
        coarse_up = coarse_fields.permute(0, 3, 1, 2)  # (B, 4, cx, cy)
        coarse_up = F.interpolate(
            coarse_up, size=(self.fine_nx, self.fine_ny),
            mode='bilinear', align_corners=True
        )  # (B, 4, fine_nx, fine_ny)
        coarse_up = coarse_up.permute(0, 2, 3, 1)  # (B, fine_nx, fine_ny, 4)
        
        # Concatenate with fine grid coordinates
        x = torch.cat([fine_coords, coarse_up], dim=-1)  # (B, fine_nx, fine_ny, 9)
        
        # Lifting
        x = self.lift(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])
        
        # Spectral blocks
        x1 = self.conv0(x); x2 = self.mlp0(x); x = F.gelu(x1 + x2)
        x1 = self.conv1(x); x2 = self.mlp1(x); x = F.gelu(x1 + x2)
        x1 = self.conv2(x); x2 = self.mlp2(x); x = F.gelu(x1 + x2)
        x1 = self.conv3(x); x2 = self.mlp3(x); x = F.gelu(x1 + x2)
        
        x = x[..., :-self.padding, :-self.padding]
        
        # Projection
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        
        return x
