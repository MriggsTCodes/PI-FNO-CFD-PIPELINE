"""
End-to-End Pipeline Orchestrator
=================================
Chains together:
  1. Coarse CFD input → FNO Super-Resolution → PINN Refinement → Final output

Provides a single forward() call for the complete pipeline,
plus access to intermediate outputs for diagnostics and ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FINE_NX, FINE_NY, COARSE_NX, COARSE_NY
from models.fno_sr import FNO_SuperResolution
from models.pinn_refiner import PINN_Refiner


class CFDSuperResPipeline(nn.Module):
    """
    Complete Coarse-CFD → FNO-SR → PINN-Refine pipeline.
    
    Stages:
      Stage 1 (FNO-SR):   Coarse solution (32×16) → Fine prediction (128×64)
      Stage 2 (PINN-Ref): Fine prediction → Physics-corrected output (128×64)
    
    The pipeline can operate in three modes:
      - 'full':     Both stages (default)
      - 'sr_only':  Only super-resolution (skip PINN refinement)
      - 'refine_only': Only PINN refinement (requires pre-computed SR output)
    
    Args:
        sr_modes: Fourier modes for the SR-FNO
        sr_width: Channel width for the SR-FNO
        refiner_hidden: Hidden width for the PINN refiner MLP
        refiner_layers: Depth of the PINN refiner MLP
    """
    
    def __init__(self, sr_modes: int = 12, sr_width: int = 48,
                 refiner_hidden: int = 128, refiner_layers: int = 6):
        super().__init__()
        
        # Stage 1: FNO Super-Resolution
        self.fno_sr = FNO_SuperResolution(
            modes1=sr_modes, modes2=sr_modes, width=sr_width,
            fine_nx=FINE_NX, fine_ny=FINE_NY
        )
        
        # Stage 2: PINN Physics Refiner
        self.pinn_refiner = PINN_Refiner(
            in_features=9,  # 5 (grid) + 4 (FNO pred)
            hidden_features=refiner_hidden,
            out_features=4,
            num_layers=refiner_layers
        )
    
    def forward(self, coarse_input: torch.Tensor, fine_coords: torch.Tensor,
                mode: str = 'full') -> dict:
        """
        Run the pipeline.
        
        Args:
            coarse_input: (B, coarse_nx, coarse_ny, 9)
                         [x, y, T_wall, U_inlet, SDF, u, v, p, T] on coarse grid
            fine_coords:  (B, fine_nx, fine_ny, 5)
                         [x, y, T_wall, U_inlet, SDF] on fine grid
            mode: 'full', 'sr_only', or 'refine_only'
            
        Returns:
            dict with keys:
              'sr_pred':     FNO super-resolution output (B, fine_nx, fine_ny, 4)
              'refined_pred': PINN-refined output (B, fine_nx, fine_ny, 4)
              'final':       The output corresponding to the selected mode
        """
        results = {}
        
        if mode in ('full', 'sr_only'):
            # Stage 1: Super-Resolution
            sr_pred = self.fno_sr(coarse_input, fine_coords)
            results['sr_pred'] = sr_pred
        
        if mode in ('full',):
            # Stage 2: Physics Refinement
            refined = self.pinn_refiner(sr_pred, fine_coords)
            results['refined_pred'] = refined
            results['final'] = refined
        elif mode == 'sr_only':
            results['final'] = sr_pred
        elif mode == 'refine_only':
            raise ValueError(
                "refine_only mode requires pre-computed SR output. "
                "Use pinn_refiner directly."
            )
        
        return results
    
    def freeze_sr(self):
        """Freeze FNO-SR weights (for Stage 2 training only)."""
        for param in self.fno_sr.parameters():
            param.requires_grad = False
        print("[Pipeline] FNO-SR weights frozen.")
    
    def unfreeze_sr(self):
        """Unfreeze FNO-SR weights."""
        for param in self.fno_sr.parameters():
            param.requires_grad = True
        print("[Pipeline] FNO-SR weights unfrozen.")
    
    def freeze_refiner(self):
        """Freeze PINN refiner weights (for Stage 1 training only)."""
        for param in self.pinn_refiner.parameters():
            param.requires_grad = False
        print("[Pipeline] PINN Refiner weights frozen.")
    
    def unfreeze_refiner(self):
        """Unfreeze PINN refiner weights."""
        for param in self.pinn_refiner.parameters():
            param.requires_grad = True
        print("[Pipeline] PINN Refiner weights unfrozen.")
    
    def count_parameters(self) -> dict:
        """Count trainable parameters in each stage."""
        sr_params = sum(p.numel() for p in self.fno_sr.parameters() if p.requires_grad)
        ref_params = sum(p.numel() for p in self.pinn_refiner.parameters() if p.requires_grad)
        total = sr_params + ref_params
        return {
            'fno_sr': sr_params,
            'pinn_refiner': ref_params,
            'total': total,
        }
