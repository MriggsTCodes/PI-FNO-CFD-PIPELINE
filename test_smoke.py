"""Quick smoke test to verify all imports and basic functionality."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *

print(f"Device: {DEVICE}")
print(f"Source dataset: {SOURCE_DATASET}")
print(f"Source dataset exists: {os.path.exists(SOURCE_DATASET)}")
print(f"Source stats exists: {os.path.exists(SOURCE_STATS)}")
print(f"Coarse grid: {COARSE_NX}x{COARSE_NY}")
print(f"Fine grid: {FINE_NX}x{FINE_NY}")
print(f"Upscale factor: {UPSCALE_FACTOR_X}x{UPSCALE_FACTOR_Y}")
print()

print("Testing data module...")
from data.geometry import make_grid, compute_sdf, fluid_mask
gx, gy = make_grid(COARSE_NX, COARSE_NY)
print(f"Coarse grid shape: {gx.shape}")
sdf = compute_sdf(gx, gy, "sine")
print(f"SDF shape: {sdf.shape}, min: {sdf.min():.6e}, max: {sdf.max():.6e}")
mask = fluid_mask(gx, gy, "sine")
print(f"Fluid mask: {mask.sum()} of {mask.size} cells ({100*mask.sum()/mask.size:.1f}%)")
print()

print("Testing model imports...")
from models.spectral_conv import SpectralConv2d
from models.fno_sr import FNO_SuperResolution
from models.pinn_refiner import PINN_Refiner
from models.pipeline import CFDSuperResPipeline
print("All model imports successful!")
print()

print("Testing model instantiation...")
import torch
model = FNO_SuperResolution(modes1=12, modes2=12, width=48)
n_params = sum(p.numel() for p in model.parameters())
print(f"FNO-SR parameters: {n_params:,}")

refiner = PINN_Refiner()
n_ref = sum(p.numel() for p in refiner.parameters())
print(f"PINN Refiner parameters: {n_ref:,}")

pipeline = CFDSuperResPipeline()
counts = pipeline.count_parameters()
print(f"Pipeline total: {counts['total']:,} params (SR: {counts['fno_sr']:,}, Ref: {counts['pinn_refiner']:,})")
print()

print("Testing forward pass (dummy data)...")
B = 1
coarse_input = torch.randn(B, COARSE_NX, COARSE_NY, 9)
fine_coords = torch.randn(B, FINE_NX, FINE_NY, 5)

sr_pred = model(coarse_input, fine_coords)
print(f"SR-FNO output shape: {sr_pred.shape}")
assert sr_pred.shape == (B, FINE_NX, FINE_NY, 4), f"Expected (1, 128, 64, 4), got {sr_pred.shape}"

refined = refiner(sr_pred, fine_coords)
print(f"PINN Refiner output shape: {refined.shape}")
assert refined.shape == (B, FINE_NX, FINE_NY, 4), f"Expected (1, 128, 64, 4), got {refined.shape}"

print()
print("=" * 60)
print("  ALL SMOKE TESTS PASSED!")
print("=" * 60)
