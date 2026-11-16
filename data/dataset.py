"""
PyTorch Dataset for Super-Resolution Training
=============================================
Wraps the paired (coarse, fine) dataset into a proper
PyTorch Dataset with train/test split logic.
"""

import os
import sys
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, TRAIN_RE_MAX, TEST_RE, DEVICE


class CyclicSRDataset(Dataset):
    """
    Dataset of (coarse_input, fine_input, fine_target) tuples
    for training the super-resolution FNO and PINN Refiner.
    
    Includes physics-preserving circular shift data augmentation
    leveraging the translation invariance of cyclic periodic channels.
    
    Coarse input: (coarse_nx, coarse_ny, 5) — [x, y, T_wall, U_inlet, SDF]
    Fine input:   (fine_nx, fine_ny, 5)     — [x, y, T_wall, U_inlet, SDF]  
    Fine target:  (fine_nx, fine_ny, 4)     — [u, v, p, T]
    """
    
    def __init__(self, paired_data: list, split: str = 'train',
                 augment: bool = True, n_shifts: int = 4,
                 cv_fold: int = None, n_cv_folds: int = 3):
        """
        Args:
            paired_data: List of dicts from coarsen.prepare_sr_dataset()
            split: 'train' (Re ≤ 60), 'test' (Re = 65), 'all', or 'cv_train' / 'cv_val'
            augment: Whether to apply cyclic shift augmentation (training only)
            n_shifts: Number of circular shifts (default 4 -> 4x data augmentation)
            cv_fold: Fold index for k-fold cross validation (0 to n_cv_folds - 1)
            n_cv_folds: Number of CV folds
        """
        self.split = split
        self.augment = augment and (split in ['train', 'cv_train'])
        self.n_shifts = n_shifts if self.augment else 1
        
        train_pool = [s for s in paired_data if s['Re'] <= TRAIN_RE_MAX]
        test_pool = [s for s in paired_data if s['Re'] == TEST_RE]
        
        if split == 'train':
            self.samples = train_pool
        elif split == 'test':
            self.samples = test_pool
        elif split == 'cv_train' and cv_fold is not None:
            # Group by Re so entire Reynolds numbers are held out
            re_values = sorted(list(set(s['Re'] for s in train_pool)))
            val_res = [re for i, re in enumerate(re_values) if i % n_cv_folds == cv_fold]
            self.samples = [s for s in train_pool if s['Re'] not in val_res]
        elif split == 'cv_val' and cv_fold is not None:
            re_values = sorted(list(set(s['Re'] for s in train_pool)))
            val_res = [re for i, re in enumerate(re_values) if i % n_cv_folds == cv_fold]
            self.samples = [s for s in train_pool if s['Re'] in val_res]
        else:
            self.samples = paired_data
            
        aug_str = f" with {self.n_shifts}x cyclic augmentation ({len(self)} effective samples)" if self.augment else ""
        print(f"[Dataset] {split} split: {len(self.samples)} base samples{aug_str}")
    
    def __len__(self):
        return len(self.samples) * self.n_shifts
    
    def __getitem__(self, idx):
        sample_idx = idx // self.n_shifts
        shift_idx = idx % self.n_shifts
        sample = self.samples[sample_idx]
        
        coarse_grid = sample['coarse_grid'].clone()  # (coarse_nx, coarse_ny, 9)
        fine_grid = sample['fine_grid'].clone()      # (fine_nx, fine_ny, 9)
        
        # Apply circular roll along x-axis if shift_idx > 0
        if shift_idx > 0:
            c_nx = coarse_grid.shape[0]
            f_nx = fine_grid.shape[0]
            c_shift = shift_idx * (c_nx // self.n_shifts)
            f_shift = shift_idx * (f_nx // self.n_shifts)
            
            # Roll geometry SDF (channel 4) and CFD fields (channels 5..8)
            # Coordinates x (ch 0), y (ch 1), Twall (ch 2), Uinlet (ch 3) remain fixed
            coarse_grid[..., 4:] = torch.roll(coarse_grid[..., 4:], shifts=c_shift, dims=0)
            fine_grid[..., 4:] = torch.roll(fine_grid[..., 4:], shifts=f_shift, dims=0)
        
        # Split into inputs and targets
        coarse_input = coarse_grid[..., 0:5]     # x, y, T_wall, U_inlet, SDF
        coarse_fields = coarse_grid[..., 5:9]    # u, v, p, T (coarse solution)
        
        fine_input = fine_grid[..., 0:5]          # x, y, T_wall, U_inlet, SDF
        fine_target = fine_grid[..., 5:9]         # u, v, p, T (ground truth)
        
        return {
            'coarse_grid': coarse_grid,           # (32, 16, 9) full paired coarse tensor
            'coarse_input': coarse_input,         # (32, 16, 5)
            'coarse_fields': coarse_fields,       # (32, 16, 4)
            'fine_input': fine_input,              # (128, 64, 5)
            'fine_target': fine_target,            # (128, 64, 4)
            'geometry': sample['geometry'],
            'Re': sample['Re'],
        }


def load_paired_dataset():
    """Load the pre-built paired dataset from disk."""
    paired_path = os.path.join(DATA_DIR, "sr_paired_dataset.pt")
    if not os.path.exists(paired_path):
        raise FileNotFoundError(
            f"Paired dataset not found at {paired_path}.\n"
            f"Run: python -m data.coarsen"
        )
    return torch.load(paired_path, weights_only=False)


def load_normalization_stats():
    """Load normalization statistics."""
    stats_path = os.path.join(DATA_DIR, "normalization_stats.pt")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Stats not found at {stats_path}.\n"
            f"Run: python -m data.coarsen"
        )
    return torch.load(stats_path, weights_only=False)
