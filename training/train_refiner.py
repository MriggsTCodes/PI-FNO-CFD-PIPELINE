"""
Train the PINN Refiner (Stage 2) — v3
======================================
Trains the physics-correction network on top of frozen FNO-SR outputs.

v3 improvements:
  - Post-FNO augmentation: circular roll applied AFTER FNO-SR forward pass
    to avoid coordinate-phase mismatch (the frozen FNO-SR always receives
    unshifted data, then sr_pred + fine_input + fine_target are rolled together)
  - Supports the new multi-scale U-Net refiner architecture
  - Best-checkpoint saving during training

Training strategy:
  1. Load a pre-trained FNO-SR model and freeze its weights
  2. For each training sample:
     a. Run FNO-SR on the UNSHIFTED sample to get sr_pred
     b. Optionally roll (sr_pred, fine_input, fine_target) together along x
     c. Run the PINN refiner to get corrected fields
     d. Compute combined data + physics + boundary + cyclic loss
  3. Only update the refiner parameters
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


def post_fno_augment(sr_pred, fine_input, fine_target, n_shifts=4):
    """
    Generate augmented (sr_pred, fine_input, fine_target) tuples by
    simultaneously rolling all three tensors along the x-axis.
    
    This avoids the coordinate-phase mismatch that occurs when rolling
    the dataset before the frozen FNO-SR forward pass. Since sin/cos
    of the linear x-coordinate are not rolled, the x-channel stays fixed
    while fields+SDF are rolled — but this is applied AFTER FNO inference,
    so the FNO always sees consistent unshifted data.
    
    For the refiner, the rolled (sr_pred, fine_input, fine_target) triplet
    is self-consistent because all three share the same shifted frame.
    
    Args:
        sr_pred:     (1, nx, ny, 4) — FNO-SR prediction
        fine_input:  (1, nx, ny, 5) — Fine grid input [x, y, Tw, Uin, SDF]
        fine_target: (1, nx, ny, 4) — Ground truth fields
        n_shifts: Number of augmented views (including shift=0)
    
    Yields:
        (sr_pred_shifted, fine_input_shifted, fine_target_shifted) tuples
    """
    nx = sr_pred.shape[1]
    
    for s in range(n_shifts):
        if s == 0:
            yield sr_pred, fine_input, fine_target
        else:
            shift = s * (nx // n_shifts)
            # Roll all tensors consistently along x (dim=1)
            sr_s = torch.roll(sr_pred, shifts=shift, dims=1)
            fi_s = fine_input.clone()
            # Roll SDF (channel 4) but keep coordinates fixed
            fi_s[..., 4:] = torch.roll(fine_input[..., 4:], shifts=shift, dims=1)
            ft_s = torch.roll(fine_target, shifts=shift, dims=1)
            yield sr_s, fi_s, ft_s


def train_refiner(sr_model_path: str = None, epochs: int = None):
    """Train the PINN Physics Refiner (Stage 2)."""
    total_epochs = epochs or REFINER_EPOCHS
    
    print("=" * 80)
    print("  PI-FNO-CFD Pipeline — Stage 2: PINN Refiner Training (v3)")
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
    
    # ── 2. Load Data (NO augmentation at dataset level) ──
    print("[Stage 2] Loading paired dataset...")
    paired_data = prepare_sr_dataset()
    train_dataset = CyclicSRDataset(
        paired_data, split='train', augment=False, n_shifts=1
    )
    
    n_aug = AUGMENT_CIRCULAR_SHIFTS
    effective_samples = len(train_dataset) * n_aug
    print(f"[Stage 2] Training dataset: {len(train_dataset)} base × {n_aug} post-FNO shifts = {effective_samples} effective samples")
    
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
    print(f"[Stage 2] PINN Refiner (U-Net) initialized: {n_params:,} parameters")
    
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
    print(f"  Post-FNO Augmentation: {n_aug}x circular shifts")
    
    # ── 5. Training Loop ──
    best_loss = float('inf')
    patience_counter = 0
    
    # Save directory (same folder as SR model)
    sr_run_folder = os.path.dirname(sr_model_path)
    best_path = os.path.join(sr_run_folder, "pinn_refiner_best.pth")
    
    print("\n[Stage 2] Starting PINN refiner training...")
    
    for epoch in range(total_epochs):
        refiner.train()
        total_data = 0.0
        total_phys = 0.0
        total_bound = 0.0
        total_cyclic = 0.0
        total_improvement = 0.0
        n_steps = 0
        
        # Annealed physics curriculum: gently ramp physics regularization
        curr_phys_weight = min(0.05, REFINER_PHYSICS_WEIGHT * (1.0 + 2.0 * (epoch / max(1, total_epochs))))
        
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            coarse_grid = sample['coarse_grid'].unsqueeze(0).to(DEVICE)
            fine_input = sample['fine_input'].unsqueeze(0).to(DEVICE)
            fine_target = sample['fine_target'].unsqueeze(0).to(DEVICE)
            
            # Get frozen SR-FNO prediction on UNSHIFTED data
            with torch.no_grad():
                sr_pred = sr_model(coarse_grid, fine_input)
            
            # Post-FNO augmentation: roll (sr_pred, fine_input, fine_target) together
            for sr_aug, fi_aug, ft_aug in post_fno_augment(sr_pred, fine_input, fine_target, n_shifts=n_aug):
                optimizer.zero_grad()
                
                # Refine the SR prediction
                refined_pred = refiner(sr_aug, fi_aug)
                
                # Loss
                losses = refiner_training_loss(
                    refined_pred, sr_aug.detach(), ft_aug, fi_aug,
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
                n_steps += 1
        
        avg_data = total_data / n_steps
        avg_phys = total_phys / n_steps
        avg_bound = total_bound / n_steps
        avg_cyclic = total_cyclic / n_steps
        avg_improvement = total_improvement / n_steps
        
        combined = (REFINER_DATA_WEIGHT * avg_data +
                    curr_phys_weight * avg_phys +
                    REFINER_BOUNDARY_WEIGHT * avg_bound +
                    REFINER_CYCLIC_WEIGHT * avg_cyclic)
        
        scheduler.step(combined)
        
        # Best checkpoint saving
        if combined < best_loss - 1e-6:
            best_loss = combined
            patience_counter = 0
            torch.save(refiner.state_dict(), best_path)
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
    
    # ── 6. Load best checkpoint and save final ──
    if os.path.exists(best_path):
        refiner.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=False))
        print(f"[Stage 2] Loaded best checkpoint (loss={best_loss:.6e})")
    
    timestamp = datetime.datetime.now().strftime("%b-%d_%I-%M-%p")
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
