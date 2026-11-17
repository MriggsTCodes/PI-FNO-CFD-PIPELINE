"""
Train the PINN Refiner (Stage 2)
================================
Trains the physics-correction network on top of frozen FNO-SR outputs.

Training strategy:
  1. Load a pre-trained FNO-SR model
  2. Freeze FNO-SR weights
  3. For each training sample:
     a. Run FNO-SR to get the super-resolution prediction
     b. Run the PINN refiner to get corrected fields
     c. Compute combined data + physics + boundary loss on corrected fields
  4. Only update the refiner parameters
"""

import os
import sys
import glob
import datetime
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DEVICE, RESULTS_DIR, DATA_DIR,
    SR_MODES, SR_WIDTH,
    REFINER_HIDDEN, REFINER_LAYERS, REFINER_EPOCHS,
    REFINER_LR, REFINER_WEIGHT_DECAY,
    REFINER_PATIENCE, REFINER_SCHEDULER_PATIENCE,
    REFINER_PHYSICS_WEIGHT, REFINER_BOUNDARY_WEIGHT, REFINER_CYCLIC_WEIGHT, REFINER_DATA_WEIGHT,
    TRAIN_RE_MAX, AUGMENT_CIRCULAR_SHIFTS
)
from models.fno_sr import FNO_SuperResolution
from models.pinn_refiner import PINN_Refiner
from data.coarsen import prepare_sr_dataset
from data.dataset import load_normalization_stats, CyclicSRDataset
from training.losses import refiner_training_loss


def find_latest_sr_model():
    """Find the most recently trained SR-FNO checkpoint."""
    pattern = os.path.join(RESULTS_DIR, "SR_FNO_*", "fno_sr_*.pth")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No SR-FNO checkpoint found. Train Stage 1 first.\n"
            f"Pattern: {pattern}"
        )
    return max(files, key=os.path.getctime)


def train_refiner(sr_model_path: str = None, epochs: int = None):
    """Train the PINN Physics Refiner (Stage 2)."""
    total_epochs = epochs or REFINER_EPOCHS
    
    print("=" * 80)
    print("  PI-FNO-CFD Pipeline — Stage 2: PINN Refiner Training")
    print("=" * 80)
    
    # ── 1. Load SR-FNO Model ──
    if sr_model_path is None:
        sr_model_path = find_latest_sr_model()
    print(f"\n[Stage 2] Loading SR-FNO from: {os.path.basename(sr_model_path)}")
    
    sr_model = FNO_SuperResolution(
        modes1=SR_MODES, modes2=SR_MODES, width=SR_WIDTH
    ).to(DEVICE)
    sr_model.load_state_dict(
        torch.load(sr_model_path, map_location=DEVICE, weights_only=False)
    )
    sr_model.eval()
    
    # Freeze SR model
    for param in sr_model.parameters():
        param.requires_grad = False
    print("[Stage 2] SR-FNO weights frozen.")
    
    # ── 2. Load Data ──
    print("[Stage 2] Loading paired dataset...")
    paired_data = prepare_sr_dataset()
    train_dataset = CyclicSRDataset(
        paired_data, split='train', augment=False, n_shifts=1
    )
    print(f"[Stage 2] Training dataset: {len(train_dataset)} augmented physical samples")
    
    stats = load_normalization_stats()
    mu = stats['mu'].to(DEVICE)
    sigma = stats['sigma'].to(DEVICE)
    mu_targets = mu[5:9]
    sigma_targets = sigma[5:9]
    
    # ── 3. Initialize Refiner ──
    refiner = PINN_Refiner(
        in_features=9,
        hidden_features=REFINER_HIDDEN,
        out_features=4,
        num_layers=REFINER_LAYERS
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in refiner.parameters())
    print(f"[Stage 2] PINN Refiner initialized: {n_params:,} parameters")
    
    # ── 4. Optimizer & Scheduler ──
    optimizer = torch.optim.Adam(
        refiner.parameters(), lr=REFINER_LR, weight_decay=REFINER_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=REFINER_SCHEDULER_PATIENCE
    )
    
    print(f"\n[Stage 2] Training Configuration:")
    print(f"  Epochs: {total_epochs}")
    print(f"  Physics Weight (base): {REFINER_PHYSICS_WEIGHT}")
    print(f"  Boundary Weight: {REFINER_BOUNDARY_WEIGHT}")
    print(f"  Cyclic Weight: {REFINER_CYCLIC_WEIGHT}")
    print(f"  Data Weight: {REFINER_DATA_WEIGHT}")
    print(f"  Augmentation: {AUGMENT_CIRCULAR_SHIFTS}x circular shifts")
    
    # ── 5. Training Loop ──
    best_loss = float('inf')
    patience_counter = 0
    
    print("\n[Stage 2] Starting PINN refiner training...")
    
    for epoch in range(total_epochs):
        refiner.train()
        total_data = 0.0
        total_phys = 0.0
        total_bound = 0.0
        total_cyclic = 0.0
        total_improvement = 0.0
        
        # Annealed physics curriculum: gently ramp physics regularization as data loss settles
        curr_phys_weight = min(0.05, REFINER_PHYSICS_WEIGHT * (1.0 + 2.0 * (epoch / max(1, total_epochs))))
        
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            coarse_grid = sample['coarse_grid'].unsqueeze(0).to(DEVICE)
            fine_input = sample['fine_input'].unsqueeze(0).to(DEVICE)
            fine_target = sample['fine_target'].unsqueeze(0).to(DEVICE)
            
            # Get frozen SR-FNO prediction
            with torch.no_grad():
                sr_pred = sr_model(coarse_grid, fine_input)
            
            optimizer.zero_grad()
            
            # Refine the SR prediction
            refined_pred = refiner(sr_pred, fine_input)
            
            # Loss
            losses = refiner_training_loss(
                refined_pred, sr_pred.detach(), fine_target, fine_input,
                mu_targets, sigma_targets,
                physics_weight=curr_phys_weight,
                boundary_weight=REFINER_BOUNDARY_WEIGHT,
                cyclic_weight=REFINER_CYCLIC_WEIGHT,
                data_weight=REFINER_DATA_WEIGHT
            )
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_data += losses['data'].item()
            total_phys += losses['physics'].item()
            total_bound += losses['boundary'].item()
            total_cyclic += losses['cyclic'].item()
            total_improvement += losses['improvement'].item()
        
        n = len(train_dataset)
        avg_data = total_data / n
        avg_phys = total_phys / n
        avg_bound = total_bound / n
        avg_cyclic = total_cyclic / n
        avg_improvement = total_improvement / n
        
        combined = (REFINER_DATA_WEIGHT * avg_data +
                    curr_phys_weight * avg_phys +
                    REFINER_BOUNDARY_WEIGHT * avg_bound +
                    REFINER_CYCLIC_WEIGHT * avg_cyclic)
        
        scheduler.step(combined)
        
        # Early stopping
        if combined < best_loss - 1e-6:
            best_loss = combined
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= REFINER_PATIENCE:
            print(f"\n[Stage 2] Early stopping at epoch {epoch}!")
            break
        
        if epoch == 1 or epoch % 25 == 0:
            lr = optimizer.param_groups[0]['lr']
            imp_pct = avg_improvement * 100
            print(f"Epoch {epoch:5d} | Data: {avg_data:.4e} | Phys: {avg_phys:.4e} | "
                  f"Bound: {avg_bound:.4e} | Cyclic: {avg_cyclic:.4e} | Improve: {imp_pct:+.2f}% | LR: {lr:.2e}", flush=True)
    
    # ── 6. Save Model ──
    timestamp = datetime.datetime.now().strftime("%b-%d_%I-%M-%p")
    
    # Save to the same run folder as the SR model
    sr_run_folder = os.path.dirname(sr_model_path)
    refiner_path = os.path.join(sr_run_folder, f"pinn_refiner_{timestamp}.pth")
    torch.save(refiner.state_dict(), refiner_path)
    print(f"\n[Stage 2] Refiner saved to {refiner_path}")
    
    return refiner, sr_run_folder


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train PINN Refiner (Stage 2)")
    parser.add_argument("--sr_model", type=str, default=None,
                        help="Path to SR-FNO checkpoint. Auto-detects latest if not specified.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    args = parser.parse_args()
    
    train_refiner(sr_model_path=args.sr_model, epochs=args.epochs)
