"""
Full Pipeline Evaluation
========================
Evaluates the complete Coarse → SR-FNO → PINN-Refine pipeline
on the held-out Re=65 zero-shot test suite.

Generates:
  1. Quantitative metrics (L2 errors, wall/bulk breakdown)
  2. Contour comparison plots (True vs Pred vs Error)
  3. Ablation comparison (Coarse-only vs SR-only vs SR+Refine)
  4. Physical benchmark quantities (pressure drop, bulk temperature)
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DEVICE, RESULTS_DIR, DATA_DIR,
    SR_MODES, SR_WIDTH,
    REFINER_HIDDEN, REFINER_LAYERS,
    FINE_NX, FINE_NY, COARSE_NX, COARSE_NY,
    TEST_RE, L, H_MIN, H_MAX, AMPLITUDE
)
from models.fno_sr import FNO_SuperResolution
from models.pinn_refiner import PINN_Refiner
from data.coarsen import prepare_sr_dataset
from data.dataset import load_normalization_stats
from data.geometry import wall_profile, fluid_mask, wall_layer_mask


FIELDS = ['Velocity X (u)', 'Velocity Y (v)', 'Pressure (p)', 'Temperature (T)']
UNITS = ['m/s', 'm/s', 'm²/s²', 'K']


def load_models(run_folder: str = None):
    """Load the trained SR-FNO and PINN refiner from a results folder."""
    
    if run_folder is None:
        # Find latest SR run folder
        sr_dirs = glob.glob(os.path.join(RESULTS_DIR, "SR_FNO_*"))
        if not sr_dirs:
            raise FileNotFoundError("No trained models found in results/")
        run_folder = max(sr_dirs, key=os.path.getctime)
    
    print(f"[Eval] Loading models from: {run_folder}")
    
    # Load SR-FNO
    sr_files = glob.glob(os.path.join(run_folder, "fno_sr_*.pth"))
    if not sr_files:
        raise FileNotFoundError(f"No SR-FNO model found in {run_folder}")
    sr_path = max(sr_files, key=os.path.getctime)
    
    sr_model = FNO_SuperResolution(
        modes1=SR_MODES, modes2=SR_MODES, width=SR_WIDTH
    ).to(DEVICE)
    sr_model.load_state_dict(
        torch.load(sr_path, map_location=DEVICE, weights_only=False)
    )
    sr_model.eval()
    print(f"  SR-FNO loaded: {os.path.basename(sr_path)}")
    
    # Load PINN refiner (optional)
    ref_files = glob.glob(os.path.join(run_folder, "pinn_refiner_*.pth"))
    refiner = None
    if ref_files:
        ref_path = max(ref_files, key=os.path.getctime)
        refiner = PINN_Refiner(
            in_features=9,
            hidden_features=REFINER_HIDDEN,
            out_features=4,
            num_layers=REFINER_LAYERS
        ).to(DEVICE)
        refiner.load_state_dict(
            torch.load(ref_path, map_location=DEVICE, weights_only=False)
        )
        refiner.eval()
        print(f"  PINN Refiner loaded: {os.path.basename(ref_path)}")
    else:
        print("  PINN Refiner not found — evaluating SR-only pipeline.")
    
    return sr_model, refiner, run_folder


def evaluate_pipeline(run_folder: str = None):
    """Full evaluation of the pipeline on the zero-shot test suite."""
    
    print("=" * 100)
    print("  PI-FNO-CFD Pipeline — Full Evaluation (Zero-Shot Re=65)")
    print("=" * 100)
    
    # Load models
    sr_model, refiner, run_folder = load_models(run_folder)
    
    # Load data
    paired_data = prepare_sr_dataset()
    test_data = [s for s in paired_data if s['Re'] == TEST_RE]
    print(f"\n[Eval] Test suite: {len(test_data)} held-out cases")
    
    # Load normalization stats
    stats = load_normalization_stats()
    mu = stats['mu'].cpu().numpy()
    sigma = stats['sigma'].cpu().numpy()
    mu_targets = mu[5:9]
    sigma_targets = sigma[5:9]
    
    # Output directory
    eval_dir = os.path.join(run_folder, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    report = []
    report.append("=" * 120)
    report.append("    PI-FNO-CFD Pipeline — Zero-Shot Generalization Report (Re = 65)")
    report.append("    Coarse (32×16) → SR-FNO (128×64) → PINN Refinement (128×64)")
    report.append("=" * 120)
    
    for sample in test_data:
        geom = sample['geometry']
        re_val = sample['Re']
        
        coarse_grid = sample['coarse_grid'].unsqueeze(0).to(DEVICE)
        fine_grid = sample['fine_grid'].unsqueeze(0).to(DEVICE)
        
        fine_input = fine_grid[..., 0:5]
        fine_target = fine_grid[..., 5:9]
        
        with torch.no_grad():
            # Stage 1: Super-Resolution
            sr_pred = sr_model(coarse_grid, fine_input)
            
            # Stage 2: PINN Refinement (if available)
            if refiner is not None:
                refined_pred = refiner(sr_pred, fine_input)
            else:
                refined_pred = sr_pred
        
        # Also compute bilinear and bicubic upsampled coarse baselines
        coarse_fields = coarse_grid[..., 5:9].permute(0, 3, 1, 2)
        bilinear_up = F.interpolate(
            coarse_fields, size=(FINE_NX, FINE_NY),
            mode='bilinear', align_corners=True
        ).permute(0, 2, 3, 1)
        bicubic_up = F.interpolate(
            coarse_fields, size=(FINE_NX, FINE_NY),
            mode='bicubic', align_corners=True
        ).permute(0, 2, 3, 1)
        
        # Convert to numpy (de-normalize)
        targets_np = fine_target.squeeze(0).cpu().numpy() * sigma_targets + mu_targets
        sr_np = sr_pred.squeeze(0).cpu().numpy() * sigma_targets + mu_targets
        refined_np = refined_pred.squeeze(0).cpu().numpy() * sigma_targets + mu_targets
        bilinear_np = bilinear_up.squeeze(0).cpu().numpy() * sigma_targets + mu_targets
        bicubic_np = bicubic_up.squeeze(0).cpu().numpy() * sigma_targets + mu_targets
        
        inputs_np = fine_input.squeeze(0).cpu().numpy()
        gx = inputs_np[..., 0]
        gy = inputs_np[..., 1]
        
        # Spatial masks
        is_fluid = fluid_mask(gx, gy, geom)
        is_wall = wall_layer_mask(gx, gy, geom, n_cells=3)
        is_bulk = is_fluid & (~is_wall)
        
        # ── Quantitative Metrics ──
        case_report = []
        case_report.append(f"\n{'─' * 145}")
        case_report.append(f"  Case: {geom.upper()} | Re = {re_val} | "
                          f"Coarse: {COARSE_NX}×{COARSE_NY} → Fine: {FINE_NX}×{FINE_NY}")
        case_report.append(f"{'─' * 145}")
        case_report.append(f"{'Field':<16} | {'Bilinear L2':>12} | {'Bicubic L2':>11} | {'SR-FNO L2':>11} | "
                          f"{'Refined L2':>11} | {'Ref MAE':>14} | {'Ref MaxErr':>14} | {'Gain':>6}")
        case_report.append("─" * 145)
        
        for i in range(4):
            true_f = targets_np[..., i]
            sr_f = sr_np[..., i]
            ref_f = refined_np[..., i]
            bil_f = bilinear_np[..., i]
            bic_f = bicubic_np[..., i]
            
            # Fluid-only metrics (true physical domain)
            true_fluid = true_f[is_fluid]
            sr_fluid = sr_f[is_fluid]
            ref_fluid = ref_f[is_fluid]
            bil_fluid = bil_f[is_fluid]
            bic_fluid = bic_f[is_fluid]
            norm_fluid = np.linalg.norm(true_fluid) + 1e-8
            
            l2_bilinear_fluid = np.linalg.norm(bil_fluid - true_fluid) / norm_fluid
            l2_bicubic_fluid = np.linalg.norm(bic_fluid - true_fluid) / norm_fluid
            l2_sr_fluid = np.linalg.norm(sr_fluid - true_fluid) / norm_fluid
            l2_refined_fluid = np.linalg.norm(ref_fluid - true_fluid) / norm_fluid
            
            # Absolute physical error metrics
            mae_ref = np.mean(np.abs(ref_fluid - true_fluid))
            max_ref = np.max(np.abs(ref_fluid - true_fluid))
            
            gain = l2_bilinear_fluid / (l2_refined_fluid + 1e-8)
            
            case_report.append(
                f"{FIELDS[i]:<16} | {l2_bilinear_fluid*100:11.2f}% | {l2_bicubic_fluid*100:10.2f}% | "
                f"{l2_sr_fluid*100:10.2f}% | {l2_refined_fluid*100:10.2f}% | "
                f"{mae_ref:11.3e} {UNITS[i]:<2} | {max_ref:11.3e} {UNITS[i]:<2} | {gain:5.1f}x"
            )
        
        # Physical benchmarks inside fluid
        p_pred = refined_np[..., 2]
        p_true = targets_np[..., 2]
        
        # Fluid-masked inlet and outlet columns
        f_inlet = is_fluid[0, :]
        f_outlet = is_fluid[-1, :]
        dp_pred = np.mean(p_pred[0, f_inlet]) - np.mean(p_pred[-1, f_outlet])
        dp_true = np.mean(p_true[0, f_inlet]) - np.mean(p_true[-1, f_outlet])
        
        # Fluid-masked bulk temperature
        u_pred = np.maximum(refined_np[..., 0], 0) * is_fluid
        T_pred = refined_np[..., 3] * is_fluid
        u_true = np.maximum(targets_np[..., 0], 0) * is_fluid
        T_true = targets_np[..., 3] * is_fluid
        
        Tb_pred = np.sum(u_pred * T_pred, axis=1) / (np.sum(u_pred, axis=1) + 1e-8)
        Tb_true = np.sum(u_true * T_true, axis=1) / (np.sum(u_true, axis=1) + 1e-8)
        
        p_std = np.std(p_true[is_fluid])
        dp_ratio = abs(dp_pred) / (p_std + 1e-8) * 100
        Tb_dev = abs(np.mean(Tb_pred) - np.mean(Tb_true))
        
        case_report.append(f"\n  Physical Benchmarks (Fluid Domain):")
        case_report.append(f"    Cyclic ΔP residual (inlet-outlet): Pred = {dp_pred:.5e} | True = {dp_true:.5e} m²/s² ({dp_ratio:.2f}% of σ_p)")
        case_report.append(f"    T_bulk (mixed-mean):             Pred = {np.mean(Tb_pred):.2f} K | True = {np.mean(Tb_true):.2f} K (Dev = {Tb_dev:.3f} K)")
        
        print("\n".join(case_report))
        report.extend(case_report)
        
        # ── Contour Plots ──
        case_dir = os.path.join(eval_dir, f"{geom}_Re{int(re_val)}")
        os.makedirs(case_dir, exist_ok=True)
        
        _plot_comparison(
            targets_np, bilinear_np, sr_np, refined_np,
            gx, gy, geom, is_fluid, case_dir, re_val
        )
    
    # Save report
    report_path = os.path.join(eval_dir, "Pipeline_Evaluation_Report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\n[Eval] Report saved to {report_path}")


def _plot_comparison(targets, bilinear, sr, refined, gx, gy, geom, is_fluid,
                     case_dir, re_val):
    """Generate comparison contour plots."""
    
    ny = gy.shape[1]
    dy = (gy[0, -1] - gy[0, 0]) / (ny - 1)
    y_lower, y_upper = wall_profile(gx[:, 0], geom)
    y_l_idx = (y_lower - gy[0, 0]) / dy
    y_u_idx = (y_upper - gy[0, 0]) / dy
    x_indices = np.arange(gx.shape[0])
    
    # 4-column comparison: True | Bilinear | SR-FNO | SR+Refine
    fig, axes = plt.subplots(4, 4, figsize=(22, 14))
    fig.suptitle(f"Pipeline Comparison — {geom.upper()} Re={int(re_val)}\n"
                 f"Coarse ({COARSE_NX}×{COARSE_NY}) → SR-FNO → PINN Refine → Fine ({FINE_NX}×{FINE_NY})",
                 fontsize=14, fontweight='bold')
    
    titles = ['CFD Truth', 'Bilinear Upsample', 'SR-FNO Output', 'SR + PINN Refined']
    data_sets = [targets, bilinear, sr, refined]
    
    for row in range(4):
        true_f = targets[..., row]
        vmin = np.min(true_f[is_fluid])
        vmax = np.max(true_f[is_fluid])
        cmap = 'coolwarm' if row == 2 else 'viridis'
        
        for col, (data, title) in enumerate(zip(data_sets, titles)):
            ax = axes[row, col]
            field = data[..., row]
            masked = np.ma.masked_where(~is_fluid, field).T
            
            ax.set_facecolor('#222222')
            im = ax.imshow(masked, origin='lower', cmap=cmap,
                          vmin=vmin, vmax=vmax, aspect='auto')
            
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight='bold')
            
            ax.plot(x_indices, y_l_idx, color='white', linestyle='--',
                   linewidth=0.8, alpha=0.7)
            ax.plot(x_indices, y_u_idx, color='white', linestyle='--',
                   linewidth=0.8, alpha=0.7)
            
            if col == 0:
                ax.set_ylabel(FIELDS[row], fontsize=10)
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    fig_path = os.path.join(case_dir, f"Pipeline_Comparison_{geom}_Re{int(re_val)}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig_path}")
    
    # Error distribution plot
    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 4))
    fig2.suptitle(f"Error Reduction — {geom.upper()} Re={int(re_val)}", fontsize=13)
    
    for i, ax in enumerate(axes2):
        true_f = targets[..., i][is_fluid]
        err_bilinear = np.abs(bilinear[..., i][is_fluid] - true_f)
        err_sr = np.abs(sr[..., i][is_fluid] - true_f)
        err_refined = np.abs(refined[..., i][is_fluid] - true_f)
        
        ax.hist(err_bilinear, bins=50, alpha=0.5, label='Bilinear', color='red', density=True)
        ax.hist(err_sr, bins=50, alpha=0.5, label='SR-FNO', color='blue', density=True)
        ax.hist(err_refined, bins=50, alpha=0.5, label='SR+Refine', color='green', density=True)
        ax.set_xlabel(f'|Error| ({UNITS[i]})')
        ax.set_ylabel('Density')
        ax.set_title(FIELDS[i])
        ax.legend(fontsize=8)
        ax.set_yscale('log')
    
    plt.tight_layout()
    fig2_path = os.path.join(case_dir, f"Error_Distribution_{geom}_Re{int(re_val)}.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {fig2_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the PI-FNO-CFD Pipeline")
    parser.add_argument("--run_folder", type=str, default=None,
                        help="Path to the results run folder. Auto-detects latest if not specified.")
    args = parser.parse_args()
    evaluate_pipeline(run_folder=args.run_folder)
