"""
Navier-Stokes Residual Computation
===================================
Computes PDE residuals using finite differences on structured grids.
Supports the full incompressible forced-convection Navier-Stokes + Energy system.

Governing Equations (dimensional, forced convection with g=0):
  Continuity:  ∂u/∂x + ∂v/∂y = 0
  X-Momentum:  u·∂u/∂x + v·∂u/∂y = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
  Y-Momentum:  u·∂v/∂x + v·∂v/∂y = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)
  Energy:      u·∂T/∂x + v·∂T/∂y = α(∂²T/∂x² + ∂²T/∂y²)

v3 Improvements:
  - Smooth SDF-based stencil confidence weighting near walls instead of hard binary cutoff
  - Prevents artificial gradient spikes from stencil points crossing the fluid-solid interface
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NU, PR, G, BETA, T_REF, ALPHA_THERMAL


def grid_gradient(f: torch.Tensor, dx: float, dy: float):
    """
    Compute spatial gradients using central finite differences.
    
    Uses torch.gradient (central differences in interior,
    forward/backward at boundaries).
    
    Args:
        f: Field tensor (batch, nx, ny)
        dx, dy: Grid spacings in x and y directions
        
    Returns:
        df_dx, df_dy: Gradient tensors (batch, nx, ny)
    """
    df_dx, df_dy = torch.gradient(f, spacing=(dx, dy), dim=(1, 2))
    return df_dx, df_dy


def compute_stencil_confidence(sdf: torch.Tensor, threshold: float = 0.02,
                                transition_width: float = 0.03) -> torch.Tensor:
    """
    Compute smooth stencil confidence weight based on SDF.
    
    Instead of a hard binary mask (SDF > threshold), this returns a smooth
    sigmoid weight that transitions from 0 (near wall) to 1 (deep interior).
    This prevents derivative stencil points near the wall interface from
    contributing large spurious residuals.
    
    For second derivatives (which use 3-point stencils), a cell at SDF=0.02
    still has stencil neighbours at SDF ≈ 0 (the wall). The smooth weight
    gracefully fades these cells' contributions rather than abruptly including
    or excluding them.
    
    Args:
        sdf: Signed distance field (batch, nx, ny)
        threshold: Center of the sigmoid transition
        transition_width: Width parameter (larger = smoother)
    
    Returns:
        weight: Smooth confidence weight in [0, 1], same shape as sdf
    """
    # sigmoid centered at threshold, with steepness controlled by transition_width
    return torch.sigmoid((sdf - threshold) / transition_width)


def navier_stokes_residuals(preds: torch.Tensor, inputs: torch.Tensor,
                            nu: float = NU, Pr: float = PR,
                            g: float = G, beta: float = BETA,
                            T_ref: float = T_REF) -> dict:
    """
    Compute all PDE residuals on the predicted fields.
    
    Args:
        preds:  (batch, nx, ny, 4) — [u, v, p, T] in physical units
        inputs: (batch, nx, ny, 5) — [x, y, T_wall, U_inlet, SDF]
        
    Returns:
        dict with keys:
          'f_u':      X-momentum residual (batch, nx, ny)
          'f_v':      Y-momentum residual (batch, nx, ny)
          'f_mass':   Continuity residual (batch, nx, ny)
          'f_T':      Energy residual (batch, nx, ny)
          'cyclic_u': Cyclic BC residual for u
          'cyclic_v': Cyclic BC residual for v
          'cyclic_p': Cyclic BC residual for p
    """
    u = preds[..., 0]
    v = preds[..., 1]
    p = preds[..., 2]
    T = preds[..., 3]
    
    x_grid = inputs[..., 0]
    y_grid = inputs[..., 1]
    
    # Grid spacing (uniform structured grid)
    dx = (x_grid[0, -1, 0] - x_grid[0, 0, 0]).item() / (x_grid.shape[1] - 1)
    dy = (y_grid[0, 0, -1] - y_grid[0, 0, 0]).item() / (y_grid.shape[2] - 1)
    
    # First derivatives
    u_x, u_y = grid_gradient(u, dx, dy)
    v_x, v_y = grid_gradient(v, dx, dy)
    p_x, p_y = grid_gradient(p, dx, dy)
    T_x, T_y = grid_gradient(T, dx, dy)
    
    # Second derivatives
    u_xx, _ = grid_gradient(u_x, dx, dy)
    _, u_yy = grid_gradient(u_y, dx, dy)
    v_xx, _ = grid_gradient(v_x, dx, dy)
    _, v_yy = grid_gradient(v_y, dx, dy)
    T_xx, _ = grid_gradient(T_x, dx, dy)
    _, T_yy = grid_gradient(T_y, dx, dy)
    
    alpha = nu / Pr
    
    # PDE Residuals (forced convection: g=0, no buoyancy)
    f_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    f_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy) - g * beta * (T - T_ref)
    f_mass = u_x + v_y
    f_T = u * T_x + v * T_y - alpha * (T_xx + T_yy)
    
    # Cyclic boundary conditions (inlet x=0 matches outlet x=L)
    cyclic_u = (u[:, 0, :] - u[:, -1, :]) ** 2
    cyclic_v = (v[:, 0, :] - v[:, -1, :]) ** 2
    cyclic_p = (p[:, 0, :] - p[:, -1, :]) ** 2
    
    return {
        'f_u': f_u,
        'f_v': f_v,
        'f_mass': f_mass,
        'f_T': f_T,
        'cyclic_u': cyclic_u,
        'cyclic_v': cyclic_v,
        'cyclic_p': cyclic_p,
    }


def physics_loss(preds: torch.Tensor, inputs: torch.Tensor,
                 mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
    """
    Scalar physics loss: Balanced MSE of all PDE residuals + cyclic BCs.
    
    Uses smooth SDF-based stencil confidence weighting instead of a hard
    binary mask. Cells near the wall interface receive lower weight, preventing
    fictitious gradient spikes from finite-difference stencils crossing the
    curved fluid-solid boundary.
    
    Terms are balanced so momentum, continuity, and energy contribute harmoniously:
      - 10.0  * f_u^2  (streamwise momentum)
      - 10.0  * f_v^2  (wall-normal momentum)
      - 0.1   * f_mass^2 (continuity)
      - 0.001 * f_T^2  (energy)
    
    Args:
        preds:  (batch, nx, ny, 4) — [u, v, p, T] in physical units
        inputs: (batch, nx, ny, 5) — [x, y, T_wall, U_inlet, SDF]
        mask:   (batch, nx, ny) or None — Boolean mask for fluid domain
        
    Returns:
        Scalar loss tensor (~0.01-0.05 on typical CFD fields)
    """
    residuals = navier_stokes_residuals(preds, inputs, **kwargs)
    
    # Smooth stencil confidence weighting
    if mask is None:
        sdf = inputs[..., 4]  # Normalized distance to wall
        # Smooth confidence: sigmoid transition from 0 (at wall) to 1 (deep interior)
        # threshold=0.03 ensures stencil cells with any neighbour in solid get low weight
        # transition_width=0.02 gives a smooth ~4-cell transition zone
        stencil_weight = compute_stencil_confidence(sdf, threshold=0.03, transition_width=0.02)
    else:
        stencil_weight = mask.float()
    
    n_effective = stencil_weight.sum().clamp(min=1.0)
    
    f_u_term = torch.sum((residuals['f_u'] ** 2) * stencil_weight) / n_effective
    f_v_term = torch.sum((residuals['f_v'] ** 2) * stencil_weight) / n_effective
    f_mass_term = torch.sum((residuals['f_mass'] ** 2) * stencil_weight) / n_effective
    f_T_term = torch.sum((residuals['f_T'] ** 2) * stencil_weight) / n_effective
    
    loss_pde = (
        10.0 * f_u_term +
        10.0 * f_v_term +
        0.1 * f_mass_term +
        0.001 * f_T_term
    )
    
    loss_cyclic = (
        torch.mean(residuals['cyclic_u']) +
        torch.mean(residuals['cyclic_v']) +
        torch.mean(residuals['cyclic_p'])
    )
    
    return loss_pde + 10.0 * loss_cyclic
