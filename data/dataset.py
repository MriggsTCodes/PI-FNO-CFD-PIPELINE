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
    for training the super-resolution FNO.
    
    Coarse input: (coarse_nx, coarse_ny, 5) — [x, y, T_wall, U_inlet, SDF]
    Fine input:   (fine_nx, fine_ny, 5)     — [x, y, T_wall, U_inlet, SDF]  
    Fine target:  (fine_nx, fine_ny, 4)     — [u, v, p, T]
    """
    
    def __init__(self, paired_data: list, split: str = 'train'):
        """
        Args:
            paired_data: List of dicts from coarsen.prepare_sr_dataset()
            split: 'train' (Re ≤ 60) or 'test' (Re = 65)
        """
        self.split = split
        
        if split == 'train':
            self.samples = [s for s in paired_data if s['Re'] <= TRAIN_RE_MAX]
        elif split == 'test':
            self.samples = [s for s in paired_data if s['Re'] == TEST_RE]
        else:
            self.samples = paired_data
            
        print(f"[Dataset] {split} split: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        coarse_grid = sample['coarse_grid']     # (coarse_nx, coarse_ny, 9)
        fine_grid = sample['fine_grid']          # (fine_nx, fine_ny, 9)
        
        # Split into inputs and targets
        coarse_input = coarse_grid[..., 0:5]     # x, y, T_wall, U_inlet, SDF
        coarse_fields = coarse_grid[..., 5:9]    # u, v, p, T (coarse solution)
        
        fine_input = fine_grid[..., 0:5]          # x, y, T_wall, U_inlet, SDF
        fine_target = fine_grid[..., 5:9]         # u, v, p, T (ground truth)
        
        return {
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
