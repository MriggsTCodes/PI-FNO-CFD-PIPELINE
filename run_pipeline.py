"""
PI-FNO-CFD Pipeline — Main Entry Point
=======================================
Orchestrates the complete workflow:
  1. Prepare paired (coarse, fine) dataset
  2. Train the Super-Resolution FNO (Stage 1)
  3. Train the PINN Physics Refiner (Stage 2)
  4. Evaluate the full pipeline on held-out test data

Usage:
  python run_pipeline.py                    # Run full pipeline
  python run_pipeline.py --stage prepare    # Only prepare data
  python run_pipeline.py --stage train_sr   # Only train SR-FNO
  python run_pipeline.py --stage train_ref  # Only train PINN refiner
  python run_pipeline.py --stage evaluate   # Only evaluate
"""

import os
import sys
import argparse
import datetime

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import DEVICE, SOURCE_DATASET


def run_prepare():
    """Stage 0: Prepare the paired dataset."""
    print("\n" + "=" * 80)
    print("  Stage 0: Preparing Paired (Coarse → Fine) Dataset")
    print("=" * 80 + "\n")
    
    from data.coarsen import prepare_sr_dataset
    prepare_sr_dataset(force_rebuild=True)


def run_train_sr(epochs=None):
    """Stage 1: Train the Super-Resolution FNO."""
    from training.train_sr import train_sr
    model, run_folder = train_sr(epochs=epochs)
    return run_folder


def run_train_refiner(sr_model_path=None, epochs=None):
    """Stage 2: Train the PINN Refiner."""
    from training.train_refiner import train_refiner
    refiner, run_folder = train_refiner(sr_model_path=sr_model_path, epochs=epochs)
    return run_folder


def run_evaluate(run_folder=None):
    """Stage 3: Evaluate the full pipeline."""
    from evaluation.evaluate import evaluate_pipeline
    evaluate_pipeline(run_folder=run_folder)


def run_full_pipeline(epochs=None):
    """Run the complete pipeline end-to-end."""
    
    print("╔" + "═" * 78 + "╗")
    print("║" + "PI-FNO-CFD Super-Resolution Pipeline".center(78) + "║")
    print("║" + "Coarse CFD → FNO Super-Resolution → PINN Physics Refinement".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    print(f"\nDevice: {DEVICE}")
    print(f"Source Dataset: {SOURCE_DATASET}")
    print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check source dataset exists
    if not os.path.exists(SOURCE_DATASET):
        print(f"\n[ERROR] Source dataset not found: {SOURCE_DATASET}")
        print("Please run the PI-FNO extract_cyclic_dataset.py first.")
        sys.exit(1)
    
    # ── Stage 0: Data Preparation ──
    run_prepare()
    
    # ── Stage 1: Train SR-FNO ──
    run_folder = run_train_sr(epochs=epochs)
    
    # Locate the saved SR model in run_folder
    import glob
    sr_files = glob.glob(os.path.join(run_folder, "fno_sr_*.pth"))
    sr_model_path = max(sr_files, key=os.path.getctime) if sr_files else None
    
    # ── Stage 2: Train PINN Refiner ──
    run_train_refiner(sr_model_path=sr_model_path, epochs=epochs)
    
    # ── Stage 3: Evaluate ──
    run_evaluate(run_folder=run_folder)
    
    print("\n" + "=" * 80)
    print(f"  Pipeline Complete! Results in: {run_folder}")
    print(f"  Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="PI-FNO-CFD Pipeline: Coarse CFD → FNO-SR → PINN Refinement",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Full pipeline
  python run_pipeline.py --stage prepare    # Data preparation only
  python run_pipeline.py --stage train_sr   # Train SR-FNO only
  python run_pipeline.py --stage train_ref  # Train PINN refiner only
  python run_pipeline.py --stage evaluate   # Evaluate only
        """
    )
    
    parser.add_argument(
        "--stage", type=str, default="full",
        choices=["full", "prepare", "train_sr", "train_ref", "evaluate"],
        help=("Which stage to run:\n"
              "  full      - Complete pipeline (default)\n"
              "  prepare   - Prepare paired dataset only\n"
              "  train_sr  - Train super-resolution FNO only\n"
              "  train_ref - Train PINN refiner only\n"
              "  evaluate  - Evaluate pipeline only")
    )
    
    parser.add_argument(
        "--sr_model", type=str, default=None,
        help="Path to a specific SR-FNO checkpoint (for train_ref or evaluate)"
    )
    
    parser.add_argument(
        "--run_folder", type=str, default=None,
        help="Path to a specific results folder (for evaluate)"
    )
    
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (overrides config defaults)"
    )
    
    args = parser.parse_args()
    
    if args.stage == "full":
        run_full_pipeline(epochs=args.epochs)
    elif args.stage == "prepare":
        run_prepare()
    elif args.stage == "train_sr":
        run_train_sr(epochs=args.epochs)
    elif args.stage == "train_ref":
        run_train_refiner(sr_model_path=args.sr_model, epochs=args.epochs)
    elif args.stage == "evaluate":
        run_evaluate(run_folder=args.run_folder)


if __name__ == "__main__":
    main()
