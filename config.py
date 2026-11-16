"""
PI-FNO-CFD Pipeline: Central Configuration
===========================================
All physical constants, geometry parameters, grid settings,
and training hyperparameters in one place.
"""

import os
import torch

# ──────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')

# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(PROJECT_ROOT)
if os.path.basename(parent_dir) == "PI-FNO":
    PI_FNO_ROOT = parent_dir
else:
    PI_FNO_ROOT = os.path.join(parent_dir, "PI-FNO")

# Source dataset (from the original PI-FNO project)
SOURCE_DATASET = os.path.join(PI_FNO_ROOT, "dataset", "cyclic_fno_dataset.pt")
SOURCE_STATS = os.path.join(PI_FNO_ROOT, "dataset", "cyclic_normalization_stats.pt")

# Pipeline-local dataset
DATA_DIR = os.path.join(PROJECT_ROOT, "data_cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ──────────────────────────────────────────────────────────
# Domain Geometry (from OpenFOAM setup_cyclic_dataset.py)
# ──────────────────────────────────────────────────────────
SCALE = 0.01
L = 1.0 * SCALE                    # Channel length [m]
H_MIN = 0.2142857 * SCALE          # Minimum channel half-height [m]
H_MAX = 0.7142857 * SCALE          # Maximum channel half-height [m]
AMPLITUDE = 0.125 * SCALE          # Wall waviness amplitude [m]

# ──────────────────────────────────────────────────────────
# Fluid Properties (Water at 308.15 K)
# ──────────────────────────────────────────────────────────
NU = 7.23e-07                      # Kinematic viscosity [m^2/s]
PR = 4.834                         # Prandtl number
BETA = 3.4e-04                     # Thermal expansion coefficient [1/K]
T_REF = 308.15                     # Reference temperature [K]
T_WALL = 293.15                    # Wall temperature [K]
G = 0.0                            # Gravitational acceleration [m/s^2] (OpenFOAM constant/g is (0,0,0) - forced convection)
ALPHA_THERMAL = NU / PR            # Thermal diffusivity [m^2/s]

# ──────────────────────────────────────────────────────────
# Grid Resolutions
# ──────────────────────────────────────────────────────────
FINE_NX, FINE_NY = 128, 64         # Fine (target) resolution
COARSE_NX, COARSE_NY = 32, 16     # Coarse (input) resolution
UPSCALE_FACTOR_X = FINE_NX // COARSE_NX   # 4x
UPSCALE_FACTOR_Y = FINE_NY // COARSE_NY   # 4x

# ──────────────────────────────────────────────────────────
# Dataset Channels (9-channel tensor format from PI-FNO)
# ──────────────────────────────────────────────────────────
# Input channels: [x, y, T_wall, U_inlet, SDF] → indices 0..4
# Output channels: [u, v, p, T] → indices 5..8
N_INPUT_CHANNELS = 5
N_OUTPUT_CHANNELS = 4
CHANNEL_NAMES = ['x', 'y', 'T_wall', 'U_inlet', 'SDF', 'u', 'v', 'p', 'T']

# ──────────────────────────────────────────────────────────
# Training Hyperparameters — Super-Resolution FNO
# ──────────────────────────────────────────────────────────
SR_MODES = 12                      # Fourier modes
SR_WIDTH = 48                      # Lifting width (wider than base FNO for SR)
SR_EPOCHS = 5000
SR_LR = 1e-3
SR_WEIGHT_DECAY = 1e-4
SR_PATIENCE = 800
SR_SCHEDULER_PATIENCE = 200
SR_PHYSICS_WEIGHT = 0.05           # Physics loss weight
SR_BOUNDARY_WEIGHT = 5.0           # Wall BC enforcement weight

# ──────────────────────────────────────────────────────────
# Training Hyperparameters — PINN Refiner
# ──────────────────────────────────────────────────────────
REFINER_HIDDEN = 64                # Conv hidden channels
REFINER_LAYERS = 4                 # Conv depth
REFINER_EPOCHS = 1000
REFINER_LR = 1e-4                  # Stable LR for convolutional refiner
REFINER_WEIGHT_DECAY = 1e-5
REFINER_PATIENCE = 300
REFINER_SCHEDULER_PATIENCE = 100
REFINER_PHYSICS_WEIGHT = 0.02      # Balanced physics regularization
REFINER_BOUNDARY_WEIGHT = 0.05     # Boundary enforcement weight
REFINER_CYCLIC_WEIGHT = 1.0        # Cyclic boundary condition enforcement weight
REFINER_DATA_WEIGHT = 1.0          # Supervised residual correction weight (primary loss)
AUGMENT_CIRCULAR_SHIFTS = 4        # Physics-preserving cyclic shifts along x (1x, 4x augmentation)

# ──────────────────────────────────────────────────────────
# SDF Blending
# ──────────────────────────────────────────────────────────
SDF_BLEND_SHARPNESS = 50.0         # tanh(C * SDF) blending sharpness

# ──────────────────────────────────────────────────────────
# Data Split
# ──────────────────────────────────────────────────────────
TRAIN_RE_MAX = 60                  # Train on Re ≤ 60
TEST_RE = 65                       # Zero-shot test on Re = 65
