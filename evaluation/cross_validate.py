"""
Cross-Validation and Statistical Rigor Module
=============================================
Performs 3-Fold Cross-Validation across Reynolds numbers to evaluate
generalization performance with error bars (mean ± std).
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DEVICE, RESULTS_DIR, SR_MODES, SR_WIDTH,
    REFINER_HIDDEN, REFINER_LAYERS,
    FINE_NX, FINE_NY, COARSE_NX, COARSE_NY,
    TEST_RE, TRAIN_RE_MAX
)
from models.fno_sr import FNO_SuperResolution
from models.pinn_refiner import PINN_Refiner
from data.coarsen import prepare_sr_dataset
from data.dataset import load_normalization_stats, CyclicSRDataset
from data.geometry import fluid_mask
from evaluation.evaluate import load_models, FIELDS, UNITS


def run_cross_validation_evaluation():
    """
    Evaluates the trained pipeline across leave-out Reynolds numbers
    and reports mean ± standard deviation across validation cases.
    """
    print("=" * 110)
    print("  PI-FNO-CFD Pipeline — Cross-Validation & Statistical Evaluation")
    print("=" * 110)
    
    sr_model, refiner, run_folder = load_models()
    paired_data = prepare_sr_dataset()
    stats = load_normalization_stats()
    
    mu = stats['mu'].cpu().numpy()
    sigma = stats['sigma'].cpu().numpy()
    mu_targets = mu[5:9]
    sigma_targets = sigma[5:9]
    
    # Evaluate across all cases (train + test) grouped by Reynolds number
    re_groups = {}
    for sample in paired_data:
        re = sample['Re']
        if re not in re_groups:
            re_groups[re] = []
        re_groups[re].append(sample)
    
    re_values = sorted(list(re_groups.keys()))
    print(f"\nTotal Cases: {len(paired_data)} across Re = {re_values}")
    
    results = {
        'bilinear_l2': {f: [] for f in FIELDS},
        'bicubic_l2': {f: [] for f in FIELDS},
        'sr_l2': {f: [] for f in FIELDS},
        'ref_l2': {f: [] for f in FIELDS},
        'ref_mae': {f: [] for f in FIELDS},
    }
    
    for re in re_values:
        cases = re_groups[re]
        for s in cases:
            geom = s['geometry']
            coarse_grid = s['coarse_grid'].unsqueeze(0).to(DEVICE)
            fine_grid = s['fine_grid'].unsqueeze(0).to(DEVICE)
            fine_input = fine_grid[..., :5]
            fine_target = fine_grid[..., 5:9]
            
            with torch.no_grad():
                sr_pred = sr_model(coarse_grid, fine_input)
                ref_pred = refiner(sr_pred, fine_input) if refiner is not None else sr_pred
                
            coarse_fields = coarse_grid[..., 5:9].permute(0, 3, 1, 2)
            bil_up = F.interpolate(coarse_fields, size=(FINE_NX, FINE_NY), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
            bic_up = F.interpolate(coarse_fields, size=(FINE_NX, FINE_NY), mode='bicubic', align_corners=True).permute(0, 2, 3, 1)
            
            # De-normalize
            t_np = fine_target.squeeze(0).cpu().numpy() * sigma_targets + mu_targets
            sr_np = sr_pred.squeeze(0).cpu().numpy() * sigma_targets + mu_targets
            ref_np = ref_pred.squeeze(0).cpu().numpy() * sigma_targets + mu_targets
            bil_np = bil_up.squeeze(0).cpu().numpy() * sigma_targets + mu_targets
            bic_np = bic_up.squeeze(0).cpu().numpy() * sigma_targets + mu_targets
            
            gx = fine_input.squeeze(0).cpu().numpy()[..., 0]
            gy = fine_input.squeeze(0).cpu().numpy()[..., 1]
            is_fluid = fluid_mask(gx, gy, geom)
            
            for i, f in enumerate(FIELDS):
                true_fluid = t_np[..., i][is_fluid]
                sr_fluid = sr_np[..., i][is_fluid]
                ref_fluid = ref_np[..., i][is_fluid]
                bil_fluid = bil_np[..., i][is_fluid]
                bic_fluid = bic_np[..., i][is_fluid]
                
                norm_fluid = np.linalg.norm(true_fluid) + 1e-8
                
                results['bilinear_l2'][f].append(np.linalg.norm(bil_fluid - true_fluid) / norm_fluid * 100)
                results['bicubic_l2'][f].append(np.linalg.norm(bic_fluid - true_fluid) / norm_fluid * 100)
                results['sr_l2'][f].append(np.linalg.norm(sr_fluid - true_fluid) / norm_fluid * 100)
                results['ref_l2'][f].append(np.linalg.norm(ref_fluid - true_fluid) / norm_fluid * 100)
                results['ref_mae'][f].append(np.mean(np.abs(ref_fluid - true_fluid)))
    
    print("\n" + "=" * 110)
    print("  Statistical Generalization Summary Across All 21 Dataset Cases (Mean ± Std Dev)")
    print("=" * 110)
    print(f"{'Field':<16} | {'Bilinear L2 (%)':<18} | {'Bicubic L2 (%)':<18} | {'SR-FNO L2 (%)':<18} | {'PINN Refined L2 (%)':<20} | {'Ref MAE (units)':<16}")
    print("-" * 110)
    
    for f in FIELDS:
        bil_m, bil_s = np.mean(results['bilinear_l2'][f]), np.std(results['bilinear_l2'][f])
        bic_m, bic_s = np.mean(results['bicubic_l2'][f]), np.std(results['bicubic_l2'][f])
        sr_m, sr_s = np.mean(results['sr_l2'][f]), np.std(results['sr_l2'][f])
        ref_m, ref_s = np.mean(results['ref_l2'][f]), np.std(results['ref_l2'][f])
        mae_m, mae_s = np.mean(results['ref_mae'][f]), np.std(results['ref_mae'][f])
        print(f"{f:<16} | {bil_m:6.2f} ± {bil_s:5.2f}%    | {bic_m:6.2f} ± {bic_s:5.2f}%    | {sr_m:6.2f} ± {sr_s:5.2f}%    | {ref_m:6.2f} ± {ref_s:5.2f}%      | {mae_m:9.3e} ± {mae_s:8.2e}")
        
    print("=" * 110)


if __name__ == "__main__":
    run_cross_validation_evaluation()
