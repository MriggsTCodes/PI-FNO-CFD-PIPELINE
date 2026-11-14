"""
Geometry utilities for the cyclic heat transfer channel.
========================================================
Computes Signed Distance Fields (SDF), wall boundaries,
and fluid masks for flat, sine, and arc channel geometries.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import L, H_MIN, H_MAX, AMPLITUDE, SCALE


def wall_profile(x: np.ndarray, geometry: str):
    """
    Compute lower and upper wall y-coordinates for a given geometry.
    
    Args:
        x: Array of x-coordinates
        geometry: One of 'flat', 'sine', 'arc'
        
    Returns:
        y_lower, y_upper: Arrays of wall y-coordinates
    """
    if geometry == "flat":
        y_lower = np.full_like(x, H_MIN / 2.0)
    elif geometry == "sine":
        y_lower = (H_MIN / 2.0) + 2.0 * AMPLITUDE * (np.sin(np.pi * x / L) ** 2)
    elif geometry == "arc":
        y_lower = (H_MIN / 2.0) + 2.0 * AMPLITUDE * np.sin(np.pi * x / L)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    
    y_upper = (H_MAX + H_MIN) - y_lower
    return y_lower, y_upper


def compute_sdf(x_grid: np.ndarray, y_grid: np.ndarray, geometry: str) -> np.ndarray:
    """
    Compute the Signed Distance Field to the nearest wall boundary.
    
    SDF = 0 at walls, positive inside the fluid domain.
    Uses a dense boundary point set + KD-tree for accurate distance queries.
    
    Args:
        x_grid: 2D array of x-coordinates (nx, ny)
        y_grid: 2D array of y-coordinates (nx, ny)
        geometry: One of 'flat', 'sine', 'arc'
        
    Returns:
        sdf: 2D array of distances to nearest wall (nx, ny)
    """
    x_bound = np.linspace(0, L, 2000)
    y_lower, y_upper = wall_profile(x_bound, geometry)
    
    lower_points = np.column_stack((x_bound, y_lower))
    upper_points = np.column_stack((x_bound, y_upper))
    all_bounds = np.vstack((lower_points, upper_points))
    
    tree = cKDTree(all_bounds)
    grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))
    distances, _ = tree.query(grid_points)
    
    return distances.reshape(x_grid.shape)


def fluid_mask(x_grid: np.ndarray, y_grid: np.ndarray, geometry: str) -> np.ndarray:
    """
    Create a boolean mask identifying fluid cells (inside the channel).
    
    Args:
        x_grid: 2D array of x-coordinates (nx, ny)
        y_grid: 2D array of y-coordinates (nx, ny)
        geometry: One of 'flat', 'sine', 'arc'
        
    Returns:
        mask: Boolean array, True for fluid cells (nx, ny)
    """
    y_lower, y_upper = wall_profile(x_grid[:, 0], geometry)
    mask = (y_grid >= y_lower[:, None]) & (y_grid <= y_upper[:, None])
    return mask


def wall_layer_mask(x_grid: np.ndarray, y_grid: np.ndarray, geometry: str,
                    n_cells: int = 3) -> np.ndarray:
    """
    Create a boolean mask for the near-wall boundary layer region.
    
    Args:
        x_grid, y_grid: 2D coordinate arrays (nx, ny)
        geometry: Channel geometry type
        n_cells: Number of grid cells from the wall to include
        
    Returns:
        mask: Boolean array, True for near-wall fluid cells
    """
    is_fluid = fluid_mask(x_grid, y_grid, geometry)
    y_lower, y_upper = wall_profile(x_grid[:, 0], geometry)
    
    dy = (y_grid[0, -1] - y_grid[0, 0]) / (y_grid.shape[1] - 1)
    wall_distance = np.minimum(
        np.abs(y_grid - y_lower[:, None]),
        np.abs(y_grid - y_upper[:, None])
    )
    
    return is_fluid & (wall_distance <= n_cells * dy)


def make_grid(nx: int, ny: int):
    """
    Create a uniform 2D mesh grid over the full domain.
    
    Args:
        nx, ny: Number of grid points
        
    Returns:
        grid_x, grid_y: 2D coordinate arrays (nx, ny)
    """
    grid_x, grid_y = np.mgrid[
        0:L:complex(0, nx),
        0:H_MAX + (H_MIN / 2.0):complex(0, ny)
    ]
    return grid_x, grid_y
