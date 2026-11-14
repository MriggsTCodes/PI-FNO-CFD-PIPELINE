"""
Train the Super-Resolution FNO (Stage 1)
========================================
Two-phase curriculum training:
  Phase 1 (Data Only):    Pure supervised fitting, no physics loss
  Phase 2 (Data+Physics): Physics-informed fine-tuning with PDE residuals

The FNO learns to map coarse CFD solutions (32×16) to fine solutions (128×64).
"""

import os
import sys
import datetime
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DEVICE, RESULTS_DIR, DATA_DIR,
    SR_MODES, SR_WIDTH, SR_EPOCHS, SR_LR, SR_WEIGHT_DECAY,
    SR_PATIENCE, SR_SCHEDULER_PATIENCE,
    SR_PHYSICS_WEIGHT, SR_BOUNDARY_WEIGHT,
    TRAIN_RE_MAX
)
from models.fno_sr import FNO_SuperResolution
from data.coarsen import prepare_sr_dataset
from data.dataset import load_normalization_stats
from training.losses import sr_training_loss


def train_sr(epochs: int = None):
    """Train the Super-Resolution FNO."""
    total_epochs = epochs or SR_EPOCHS
    
    print("=" * 80)
    print("  PI-FNO-CFD Pipeline — Stage 1: Super-Resolution FNO Training")
    print("=" * 80)
    
    # ── 1. Load Data ──
    print("\n[Stage 1] Preparing paired dataset...")
    paired_data = prepare_sr_dataset()
    
    train_data = [s for s in paired_data if s['Re'] <= TRAIN_RE_MAX]
    print(f"[Stage 1] Training on {len(train_data)} samples (Re ≤ {TRAIN_RE_MAX})")
    
    # Load normalization stats
    stats = load_normalization_stats()
    mu = stats['mu'].to(DEVICE)
    sigma = stats['sigma'].to(DEVICE)
    mu_targets = mu[5:9]
    sigma_targets = sigma[5:9]
    
    # ── 2. Initialize Model ──
    model = FNO_SuperResolution(
        modes1=SR_MODES, modes2=SR_MODES, width=SR_WIDTH
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Stage 1] FNO-SR model initialized: {n_params:,} parameters")
    
    # ── 3. Optimizer & Scheduler ──
    optimizer = torch.optim.Adam(
        model.parameters(), lr=SR_LR, weight_decay=SR_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=SR_SCHEDULER_PATIENCE
    )
    
    # ── 4. Training Configuration ──
    stage1_epochs = max(1, total_epochs // 3)  # First third: data only
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"\n[Stage 1] Training Configuration:")
    print(f"  Total Epochs:           {total_epochs}")
    print(f"  Phase 1 (Data Only):    Epochs 0 - {stage1_epochs - 1}")
    print(f"  Phase 2 (Data+Physics): Epochs {stage1_epochs} - {total_epochs - 1}")
    print(f"  Physics Weight: {SR_PHYSICS_WEIGHT}")
    print(f"  Boundary Weight: {SR_BOUNDARY_WEIGHT}")
    print(f"  Device: {DEVICE}")
    
    # ── 5. Training Loop ──
    print("\n[Stage 1] Starting training...")
    
    for epoch in range(total_epochs):
        model.train()
        total_data = 0.0
        total_phys = 0.0
        total_bound = 0.0
        
        in_phase2 = epoch >= stage1_epochs
        
        for sample in train_data:
            coarse_grid = sample['coarse_grid'].unsqueeze(0).to(DEVICE)  # (1, cx, cy, 9)
            fine_grid = sample['fine_grid'].unsqueeze(0).to(DEVICE)      # (1, fx, fy, 9)
            
            fine_input = fine_grid[..., 0:5]
            fine_target = fine_grid[..., 5:9]
            
            optimizer.zero_grad()
            
            # Forward pass
            sr_pred = model(coarse_grid, fine_input)
            
            # Loss
            losses = sr_training_loss(
                sr_pred, fine_target, fine_input,
                mu_targets, sigma_targets,
                physics_weight=SR_PHYSICS_WEIGHT,
                boundary_weight=SR_BOUNDARY_WEIGHT,
                use_physics=in_phase2
            )
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_data += losses['data'].item()
            total_phys += losses['physics'].item()
            total_bound += losses['boundary'].item()
        
        n = len(train_data)
        avg_data = total_data / n
        avg_phys = total_phys / n
        avg_bound = total_bound / n
        
        current_phys_w = SR_PHYSICS_WEIGHT if in_phase2 else 0.0
        combined = avg_data + current_phys_w * avg_phys + SR_BOUNDARY_WEIGHT * avg_bound
        
        scheduler.step(combined)
        
        # Early stopping (only in phase 2)
        if in_phase2:
            if combined < best_loss - 1e-6:
                best_loss = combined
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= SR_PATIENCE:
                print(f"\n[Stage 1] Early stopping at epoch {epoch}!")
                break
        
        # Reset for phase 2 transition
        if epoch == stage1_epochs:
            print(f"\n{'=' * 80}")
            print(f">>> PHASE 2 ACTIVATED: Physics loss engaged (weight={SR_PHYSICS_WEIGHT})")
            print(f"{'=' * 80}\n")
            best_loss = float('inf')
            patience_counter = 0
        
        if epoch % 25 == 0:
            phase = "P1-Data" if not in_phase2 else "P2-Phys"
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:5d} [{phase}] | Data: {avg_data:.4e} | "
                  f"Phys: {avg_phys:.4e} | Bound: {avg_bound:.4e} | LR: {lr:.2e}")
    
    # ── 6. Save Model ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%b-%d_%I-%M-%p")
    run_folder = os.path.join(RESULTS_DIR, f"SR_FNO_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    
    model_path = os.path.join(run_folder, f"fno_sr_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n[Stage 1] Model saved to {model_path}")
    
    return model, run_folder


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Super-Resolution FNO (Stage 1)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    args = parser.parse_args()
    train_sr(epochs=args.epochs)
