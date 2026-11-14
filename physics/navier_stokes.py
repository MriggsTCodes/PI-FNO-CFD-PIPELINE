"""
Navier-Stokes Residual Computation
===================================
Computes PDE residuals using finite differences on structured grids.
Supports the full incompressible Boussinesq Navier-Stokes + Energy system.

Governing Equations (dimensional):
  Continuity:  ∂u/∂x + ∂v/∂y = 0
  X-Momentum:  u·∂u/∂x + v·∂u/∂y = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
  Y-Momentum:  u·∂v/∂x + v·∂v/∂y = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²) + gβ(T - T_ref)
  Energy:      u·∂T/∂x + v·∂T/∂y = α(∂²T/∂x² + ∂²T/∂y²)
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
    
    # PDE Residuals
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
    
    Residuals are evaluated strictly on fluid interior cells (away from wall interface)
    to prevent spurious boundary finite-difference artifacts.
    Terms are balanced so momentum, continuity, and energy contribute harmoniously:
      - 10.0  * f_u^2
      - 1.0   * f_v^2
      - 0.1   * f_mass^2
      - 0.001 * f_T^2
    
    Args:
        preds:  (batch, nx, ny, 4) — [u, v, p, T] in physical units
        inputs: (batch, nx, ny, 5) — [x, y, T_wall, U_inlet, SDF]
        mask:   (batch, nx, ny) or None — Boolean mask for fluid domain
        
    Returns:
        Scalar loss tensor (~0.01-0.05 on typical CFD fields)
    """
    residuals = navier_stokes_residuals(preds, inputs, **kwargs)
    
    # Identify fluid domain: use SDF if mask is not explicitly provided
    if mask is None:
        sdf = inputs[..., 4]  # Normalized distance to wall
        # Fluid interior: cells away from the immediate wall interface (SDF > 0.02)
        fluid_interior = (sdf > 0.02).float()
    else:
        fluid_interior = mask.float()
    
    n_fluid = fluid_interior.sum().clamp(min=1.0)
    
    f_u_term = torch.sum((residuals['f_u'] ** 2) * fluid_interior) / n_fluid
    f_v_term = torch.sum((residuals['f_v'] ** 2) * fluid_interior) / n_fluid
    f_mass_term = torch.sum((residuals['f_mass'] ** 2) * fluid_interior) / n_fluid
    f_T_term = torch.sum((residuals['f_T'] ** 2) * fluid_interior) / n_fluid
    
    loss_pde = (
        10.0 * f_u_term +
        1.0 * f_v_term +
        0.1 * f_mass_term +
        0.001 * f_T_term
    )
    
    loss_cyclic = (
        torch.mean(residuals['cyclic_u']) +
        torch.mean(residuals['cyclic_v']) +
        torch.mean(residuals['cyclic_p'])
    )
    
    return loss_pde + 10.0 * loss_cyclic
