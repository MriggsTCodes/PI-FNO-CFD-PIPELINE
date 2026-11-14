"""
Coarsening & Super-Resolution Data Preparation
===============================================
Takes the existing fine-resolution (128×64) cyclic CFD dataset
and creates paired (coarse, fine) samples for training the
super-resolution FNO.

Coarsening Strategy:
  - Subsample the fine grid by the upscale factor (4× in each dim)
  - Recompute SDF on the coarse grid for geometric consistency
  - Recompute coordinate grids on the coarse mesh
  - Preserve the same normalization scheme
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SOURCE_DATASET, SOURCE_STATS, DATA_DIR,
    FINE_NX, FINE_NY, COARSE_NX, COARSE_NY,
    UPSCALE_FACTOR_X, UPSCALE_FACTOR_Y,
    L, H_MIN, H_MAX, AMPLITUDE, SCALE, NU
)
from data.geometry import compute_sdf, make_grid


def coarsen_grid(fine_grid: torch.Tensor, geometry: str,
                 coarse_nx: int = COARSE_NX, coarse_ny: int = COARSE_NY) -> torch.Tensor:
    """
    Downsample a fine-resolution grid tensor to a coarse resolution.
    
    For coordinate and SDF channels: recompute exactly on the coarse mesh.
    For field channels (u, v, p, T) and BCs (T_wall, U_inlet): area-average
    using bilinear interpolation to simulate a coarse CFD solution.
    
    Args:
        fine_grid: Tensor of shape (fine_nx, fine_ny, 9)
        geometry: 'flat', 'sine', or 'arc'
        coarse_nx, coarse_ny: Target coarse resolution
        
    Returns:
        coarse_grid: Tensor of shape (coarse_nx, coarse_ny, 9)
    """
    fine_nx, fine_ny, n_ch = fine_grid.shape
    
    # Recompute coordinates on the coarse grid
    cx, cy = make_grid(coarse_nx, coarse_ny)
    
    # Recompute SDF on the coarse grid (geometric consistency)
    sdf_coarse = compute_sdf(cx, cy, geometry)
    
    # Downsample field channels via bilinear interpolation
    # Reshape to (1, channels, nx, ny) for F.interpolate
    fields = fine_grid.permute(2, 0, 1).unsqueeze(0).float()  # (1, 9, nx, ny)
    
    coarse_fields = F.interpolate(
        fields, size=(coarse_nx, coarse_ny), mode='bilinear', align_corners=True
    )  # (1, 9, coarse_nx, coarse_ny)
    
    coarse_grid = coarse_fields.squeeze(0).permute(1, 2, 0)  # (coarse_nx, coarse_ny, 9)
    
    # Overwrite coordinates and SDF with exact coarse-grid values
    coarse_grid[..., 0] = torch.tensor(cx, dtype=torch.float32)
    coarse_grid[..., 1] = torch.tensor(cy, dtype=torch.float32)
    # Channels 2, 3 (T_wall, U_inlet) are uniform → kept from interpolation
    coarse_grid[..., 4] = torch.tensor(sdf_coarse, dtype=torch.float32)
    
    return coarse_grid


def prepare_sr_dataset(force_rebuild: bool = False):
    """
    Build paired (coarse, fine) dataset from the existing cyclic FNO dataset.
    
    Each sample contains:
      - 'coarse_grid': Tensor (coarse_nx, coarse_ny, 9) — coarsened input
      - 'fine_grid':   Tensor (fine_nx, fine_ny, 9) — original fine target
      - 'geometry':    str
      - 'Re':          float
      - 'U_inlet':     float
      - 'T_wall':      float
      
    Saves the paired dataset and normalization stats to data_cache/.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    paired_path = os.path.join(DATA_DIR, "sr_paired_dataset.pt")
    if os.path.exists(paired_path) and not force_rebuild:
        print(f"[Coarsen] Paired dataset already exists at {paired_path}")
        return torch.load(paired_path, weights_only=False)
    
    print(f"[Coarsen] Loading source dataset from {SOURCE_DATASET}...")
    if not os.path.exists(SOURCE_DATASET):
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_DATASET}\n"
            f"Run the PI-FNO extract_cyclic_dataset.py first."
        )
    
    dataset = torch.load(SOURCE_DATASET, weights_only=False)
    print(f"[Coarsen] Loaded {len(dataset)} samples from source dataset.")
    
    # Load normalization stats
    stats = None
    if os.path.exists(SOURCE_STATS):
        stats = torch.load(SOURCE_STATS, weights_only=False)
        print(f"[Coarsen] Loaded normalization stats.")
    
    paired_dataset = []
    for i, sample in enumerate(dataset):
        geometry = sample['geometry']
        Re = sample['Re']
        fine_grid = sample['grid']  # (128, 64, 9) — already normalized
        
        print(f"  [{i+1}/{len(dataset)}] Coarsening {geometry}_Re{int(Re)} "
              f"({fine_grid.shape[0]}×{fine_grid.shape[1]} → {COARSE_NX}×{COARSE_NY})...")
        
        # We need the un-normalized SDF for correct coarse recomputation
        # The SDF in fine_grid[..., 4] is min-max normalized [0,1]
        # For coarsening, we recompute SDF on the coarse grid and then
        # apply the same min-max normalization
        coarse_grid = coarsen_grid(fine_grid, geometry)
        
        # Apply the same SDF min-max normalization as the fine grid
        # (the coarse SDF was computed raw, but the fine SDF was already normalized)
        # Since we overwrite SDF with raw values, we need to normalize:
        # We'll handle this properly by using the fine grid's SDF range
        
        paired_sample = {
            'coarse_grid': coarse_grid,
            'fine_grid': fine_grid,
            'geometry': geometry,
            'Re': Re,
            'U_inlet': sample.get('U_inlet', 0.0),
            'T_wall': sample.get('T_wall', 293.15),
            'mesh_coarse': (COARSE_NX, COARSE_NY),
            'mesh_fine': (FINE_NX, FINE_NY),
        }
        paired_dataset.append(paired_sample)
    
    # Normalize the coarse SDF consistently with fine SDF
    # Fine SDF was min-max normalized globally. We need to do the same for coarse.
    all_coarse_sdf = torch.cat([s['coarse_grid'][..., 4].flatten() for s in paired_dataset])
    sdf_min = all_coarse_sdf.min()
    sdf_max = all_coarse_sdf.max()
    sdf_range = sdf_max - sdf_min
    if sdf_range < 1e-10:
        sdf_range = torch.tensor(1.0)
    
    for s in paired_dataset:
        s['coarse_grid'][..., 4] = (s['coarse_grid'][..., 4] - sdf_min) / sdf_range
    
    # Save the coarse SDF normalization stats alongside the dataset
    coarse_sdf_stats = {'sdf_min': sdf_min, 'sdf_max': sdf_max}
    
    torch.save(paired_dataset, paired_path)
    print(f"\n[Coarsen] Saved {len(paired_dataset)} paired samples to {paired_path}")
    
    # Copy normalization stats to local cache
    if stats is not None:
        stats_local = os.path.join(DATA_DIR, "normalization_stats.pt")
        stats['coarse_sdf_min'] = sdf_min
        stats['coarse_sdf_max'] = sdf_max
        torch.save(stats, stats_local)
        print(f"[Coarsen] Saved normalization stats to {stats_local}")
    
    return paired_dataset


if __name__ == "__main__":
    prepare_sr_dataset(force_rebuild=True)
