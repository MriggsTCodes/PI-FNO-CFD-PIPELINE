# PI-FNO-CFD Pipeline

## Accelerating CFD with Physics-Informed Neural Operator Super-Resolution

A framework that combines **coarse-mesh CFD**, **Fourier Neural Operator (FNO) super-resolution**, and **Physics-Informed Neural Network (PINN) refinement** to accelerate computational fluid dynamics simulations while maintaining physical accuracy.

### Core Idea

Instead of running expensive fine-mesh CFD simulations, we:

1. **Run CFD on a coarse mesh** (32×16) — ~16× faster than fine mesh
2. **Upscale with FNO** (32×16 → 128×64) — learns the coarse-to-fine mapping in Fourier space
3. **Refine with PINN** — physics-informed correction enforces Navier-Stokes, continuity, energy equations, and wall boundary conditions

```
OpenFOAM CFD      FNO Super-         PINN Physics        High-Resolution
(32×16 mesh)  →   Resolution    →    Refinement    →     Solution
~0.06x cost       (learned)          (physics)           (128×64 mesh)
```

### Architecture

#### Stage 1: FNO Super-Resolution
- **Input**: Coarse CFD fields (u, v, p, T) + grid coordinates + SDF + BCs
- **Processing**: 4 spectral conv layers at coarse resolution → bilinear upsampling → 2 spectral conv layers at fine resolution
- **Output**: High-resolution field predictions
- **Key insight**: FNO's spectral layers are resolution-invariant — same Fourier modes work at any grid size

#### Stage 2: PINN Physics Refinement
- **Input**: FNO super-resolution output + grid coordinates & SDF
- **Processing**: 4-layer Cyclic 2D Residual Convolutional Network (`CyclicConv2d`) with stream-wise $x$-periodicity, wall-replicate $y$-padding, group normalization, and zero-initialized output layer.
- **Physics Regularization**: Fluid-interior masked incompressible Navier-Stokes, continuity, and energy equation residuals with balanced dimensionless loss scaling.
- **Output**: Physics-corrected fields satisfying PDE constraints and wall boundary conditions.
- **Key insight**: Spatial convolutions preserve continuous derivatives; zero-initialization guarantees strictly non-negative refinement gains over FNO.

### Dataset

Uses the cyclic heat transfer channel dataset with:
- **3 geometries**: flat, sine-wave, arc-wall channels
- **Training**: Re = 55, 56, 57, 58, 59, 60 (18 samples)
- **Zero-shot test**: Re = 65 (3 samples, never seen during training)
- **Fluid**: Water at 308.15 K (ν = 7.23×10⁻⁷, Pr = 4.834)

### Zero-Shot Generalization Benchmark ($Re = 65$)

| Geometry | Field | Bilinear Baseline | Stage 1 (SR-FNO) | Stage 2 (PINN Refined) | Gain vs Bilinear |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Arc Channel** | Velocity $u$ | 6.58% | 3.27% | **2.34%** | **2.8× lower error** |
| | Velocity $v$ | 16.23% | 5.42% | **4.98%** | **3.3× lower error** |
| | Pressure $p$ | 6.48% | 3.10% | **2.79%** | **2.3× lower error** |
| **Sine Channel** | Velocity $u$ | 5.89% | 4.46% | **3.47%** | **1.7× lower error** |
| | Velocity $v$ | 16.86% | 5.70% | **5.26%** | **3.2× lower error** |
| | Pressure $p$ | 6.75% | 3.87% | **3.56%** | **1.9× lower error** |
| **Flat Channel** | Velocity $u$ | 7.77% | 5.97% | **4.54%** | **1.7× lower error** |

### Usage

```bash
# Activate the virtual environment
source ../venv/bin/activate

# Run the full pipeline
python run_pipeline.py

# Or run individual stages
python run_pipeline.py --stage prepare      # Prepare paired dataset
python run_pipeline.py --stage train_sr     # Train SR-FNO (Stage 1)
python run_pipeline.py --stage train_ref    # Train PINN refiner (Stage 2)
python run_pipeline.py --stage evaluate     # Evaluate pipeline
```

### Project Structure

```
PI-FNO-CFD-Pipeline/
├── config.py                 # All constants, hyperparameters, paths
├── run_pipeline.py           # Main entry point
├── data/
│   ├── geometry.py           # Wall profiles, SDF, fluid masks
│   ├── coarsen.py            # Create paired (coarse, fine) training data
│   └── dataset.py            # PyTorch Dataset with train/test split
├── models/
│   ├── spectral_conv.py      # Fourier-space convolution layer
│   ├── fno_sr.py             # FNO super-resolution model
│   ├── pinn_refiner.py       # PINN physics correction network
│   └── pipeline.py           # End-to-end pipeline orchestrator
├── physics/
│   ├── navier_stokes.py      # PDE residual computation
│   └── boundary.py           # Boundary condition enforcement
├── training/
│   ├── losses.py             # Combined loss functions
│   ├── train_sr.py           # Stage 1 training (SR-FNO)
│   └── train_refiner.py      # Stage 2 training (PINN refiner)
├── evaluation/
│   └── evaluate.py           # Full pipeline evaluation + plots
├── data_cache/               # Cached datasets (auto-generated)
└── results/                  # Training outputs (auto-generated)
```

### Dependencies

Uses the same virtual environment as the parent PI-FNO project. Key dependencies:
- PyTorch ≥ 2.5
- NumPy, SciPy
- Matplotlib
