"""
Spectral Convolution Layer
==========================
2D spectral convolution in Fourier space (core building block of FNO).
Adapted from the original PI-FNO implementation with support for
arbitrary input/output resolutions (key for super-resolution).
"""

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """
    2D Fourier-space convolution layer.
    
    Learns weights on the low-frequency Fourier modes and multiplies
    them with the input's FFT. Captures global spatial structure
    that local convolutions miss.
    
    Args:
        in_channels: Number of input feature channels
        out_channels: Number of output feature channels
        modes1: Number of Fourier modes in the x-direction
        modes2: Number of Fourier modes in the y-direction
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2,
                              dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2,
                              dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex matrix multiplication in Fourier space.
        
        (batch, in_ch, x, y) × (in_ch, out_ch, x, y) → (batch, out_ch, x, y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, nx, ny)
            
        Returns:
            (batch, out_channels, nx, ny)
        """
        batchsize = x.shape[0]
        
        # Forward FFT (real → complex)
        x_ft = torch.fft.rfft2(x)
        
        # Allocate output in Fourier space
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Determine valid mode bounds for current grid resolution
        m1 = min(self.modes1, x_ft.size(-2) // 2)
        m2 = min(self.modes2, x_ft.size(-1))
        
        # Multiply low-frequency modes with learned weights
        # Top-left corner (positive frequencies)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        # Bottom-left corner (negative frequencies, wrapped)
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )
        
        # Inverse FFT back to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
