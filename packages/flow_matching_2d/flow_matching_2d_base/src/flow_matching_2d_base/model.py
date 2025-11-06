# -*- coding: utf-8 -*-
# packages/flow_matching_2d/flow_matching_2d_base/src/flow_matching_2d_base/model.py
#
# --- MODIFIED ---
# This model file combines the UNet architecture from the user-provided 'unet.py'
# with the 'PaddedConv2D' (physics-informed padding) from the user's
# 'low_rank_tckae_2d' model.
#
# Key components:
# 1. PaddedConv2D: The user's custom Conv2D layer with physics-padding.
# 2. MaxPool, DoubleConv, ResNetBlock, UNet: Adapted from 'unet.py' to use
#    PaddedConv2D and remove all generic padding arguments.
# 3. SinusoidalTimeEmbedding: Standard module for flow/diffusion models.
# 4. FlowMatchingUNet: The top-level wrapper that combines the UNet with
#    time embeddings and conditional inputs.
#
# --- FIX (11-06-2025) ---
# Added 'F.pad' logic in the UNet 'forward' method to fix a RuntimeError
# caused by odd spatial dimensions (e.g., 127) leading to a mismatch
# in the skip connections (e.g., size 63 vs 62).
# --- END FIX ---

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# --- CUSTOM MODULE FOR PHYSICS-INFORMED PADDING ---
# From user's low_rank_tckae_2d model.py
class PaddedConv2D(nn.Module):
    """
    A custom 2D convolutional layer that applies physics-informed padding before
    the convolution operation. This is designed for the specific flow problem:
    - Z-axis (Top/Bottom): Zero-padding for no-slip walls.
    - Max X-axis (Right): Reflection-padding for the outlet.
    - Min X-axis (Left): Replication-padding for the inlet.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        # For a kernel_size of 3, the padding amount is 1.
        self.padding_amount = (kernel_size - 1) // 2

        # The actual convolution layer has no padding, as we handle it manually.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0  # IMPORTANT: Manual padding is done in the forward pass.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies padding sequentially to each boundary based on its physics.
        The padding tuple for F.pad is (pad_left, pad_right, pad_top, pad_bottom).
        """
        p = self.padding_amount
        
        # 1. Pad Z-axis (Top/Bottom) with Zeros for the no-slip condition.
        # Pads by `p` on the top and `p` on the bottom.
        padded_x = F.pad(x, (0, 0, p, p), mode='constant', value=0)

        # 2. Pad Max X-axis (Right) with Reflection for the outlet condition.
        # Pads by `p` on the right side.
        padded_x = F.pad(padded_x, (0, p, 0, 0), mode='reflect')

        # 3. Pad Min X-axis (Left) with Replication for the inlet condition.
        # Pads by `p` on the left side.
        padded_x = F.pad(padded_x, (p, 0, 0, 0), mode='replicate')

        # Now, apply the convolution to the correctly padded tensor.
        return self.conv(padded_x)


# --- UNET BUILDING BLOCKS (Adapted from user's unet.py) ---

class MaxPool(nn.Module):
    """
    Wrapper for 1D/2D/3D Max Pooling.
    (From user's unet.py)
    """
    def __init__(self, *ks):
        super().__init__()
        if len(ks) == 1:   self.mp = nn.MaxPool1d(ks, ks)
        elif len(ks) == 2: self.mp = nn.MaxPool2d(ks, ks)
        elif len(ks) == 3: self.mp = nn.MaxPool3d(ks, ks)
        else: raise Exception("Invalid number of dimensions for MaxPool.")

    def forward(self, x):
        return self.mp(x)


class DoubleConv(nn.Module):
    """
    Double Convolution block (Conv -> Norm -> NL -> Conv).
    
    --- MODIFIED ---
    - Uses PaddedConv2D instead of nn.Conv2d.
    - All 'padding' arguments removed as padding is now hard-coded.
    - Uses nn.InstanceNorm2d as seen in unet.py.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dims=2, # Kept for potential 3D extension, though PaddedConv2D is 2D
                 nl=nn.GELU(), # Defaulted to GELU
                 include_norm=True):
        super().__init__()
        
        if dims != 2:
            raise ValueError(f"PaddedConv2D is only for dims=2, but got dims={dims}")

        self.norm = None
        if include_norm:
            # Using InstanceNorm as in the original unet.py
            self.norm = nn.InstanceNorm2d(num_features=out_channels, eps=1.0)

        self.nl = nl
        
        # --- MODIFIED: Use PaddedConv2D ---
        # Note: Stride=1, Kernel=3 is standard for UNet DoubleConv
        self.conv1 = PaddedConv2D(in_channels, out_channels, kernel_size=3, stride=1)
        self.conv2 = PaddedConv2D(out_channels, out_channels, kernel_size=3, stride=1)
        # --- END MODIFICATION ---

    def forward(self, x):
        x = self.conv1(x)
        if self.norm is not None: x = self.norm(x)
        x = self.nl(x)
        x = self.conv2(x)
        return x


class ResNetBlock(nn.Module):
    """
    ResNet block (Input -> DoubleConv) + (Input -> 1x1 Conv).
    
    --- MODIFIED ---
    - 'padding' arguments removed.
    - Manual padding in forward() removed, as DoubleConv now handles it.
    - Uses nn.Conv2d for the 1x1 kernel (which doesn't need padding).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dims=2,
                 nl=nn.GELU(), # Defaulted to GELU
                 include_norm=True):
        super().__init__()
        
        if dims != 2:
            raise ValueError(f"PaddedConv2D is only for dims=2, but got dims={dims}")
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dims = dims

        self.dc = DoubleConv(in_channels, out_channels, dims=dims, nl=nl, include_norm=include_norm)
        
        # 1x1 Conv for the residual connection
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # --- MODIFIED: Removed manual padding ---
        # The DoubleConv 'self.dc' now handles its own padding internally.
        z = self.dc(x)
        
        # Residual 1x1 conv path
        y = self.conv(x)
        # --- END MODIFICATION ---

        return y + z


class UNet(nn.Module):
    """
    UNet architecture.
    
    --- MODIFIED ---
    - 'padding' arguments removed.
    - 'nl' (non-linearity) defaulted to GELU.
    - **FIX:** Added F.pad in forward() to handle odd-sized inputs.
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 features=(32, 64),
                 nl=nn.GELU()): # Defaulted to GELU
        super().__init__()
        self.dims = 2 # Hard-coded for 2D
            
        self.encoders = torch.nn.ModuleList([])
        self.bridges  = torch.nn.ModuleList([])
        self.decoders = torch.nn.ModuleList([])
        self.up = nn.Upsample(scale_factor=tuple([2]*self.dims), mode="nearest")
        self.pool = MaxPool(*[2]*self.dims)
        self.nl = nl

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_top_unet_block(in_channels, features[0], out_channels)
        for k in range(len(features)-1):
            self.append_unet_block(features[k+1])

    def forward(self, x):
        x_bridged = []
        for k in range(len(self.encoders)):
            x = self.encoders[k](x)
            x_bridged.append( self.bridges[k](x) )
            if k < len(self.encoders) - 1: x = self.pool(x)
        
        x_up = None # Initialize
        
        for k in range(len(self.decoders)-1, -1, -1):
            if k == len(self.decoders)-1:
                # At the bottom of the U, start with zeros
                x_up = torch.zeros_like( x_bridged[k] )
            
            # --- FIX for odd-sized inputs ---
            # Pad x_up to match x_bridged[k] if sizes differ
            # This handles cases where max-pooling (e.g., 63 -> 31)
            # followed by upsampling (e.g., 31 -> 62)
            # results in a size mismatch with the skip connection (e.g., 63).
            if x_up.shape != x_bridged[k].shape:
                # B, C, H, W
                target_shape = x_bridged[k].shape
                # (pad_left, pad_right, pad_top, pad_bottom)
                # We pad on the right and bottom, assuming dims[2]=H, dims[3]=W
                padding_dims = (
                    0, target_shape[3] - x_up.shape[3], # W
                    0, target_shape[2] - x_up.shape[2]  # H
                )
                x_up = F.pad(x_up, padding_dims, "constant", 0)
            # --- END FIX ---
            
            x = self.decoders[k]( x_bridged[k] + x_up )
            
            if k > 0:
                x_up = self.up(x)
                
        return x

    def add_top_unet_block(self, in_channels, bridge_channels, out_channels):
        encoder = ResNetBlock(in_channels,     bridge_channels, nl=self.nl, dims=self.dims, include_norm=False)
        bridge  = ResNetBlock(bridge_channels, bridge_channels, nl=self.nl, dims=self.dims, include_norm=False)
        decoder = ResNetBlock(bridge_channels, out_channels,    nl=self.nl, dims=self.dims, include_norm=False)

        self.encoders.append( encoder )
        self.bridges.append(  bridge )
        self.decoders.append( decoder )
    
    def append_unet_block(self, bridge_channels):
        top_channels = self.encoders[-1].out_channels

        encoder = ResNetBlock(top_channels,    bridge_channels, nl=self.nl, dims=self.dims, include_norm=False)
        bridge  = ResNetBlock(bridge_channels, bridge_channels, nl=self.nl, dims=self.dims, include_norm=False)
        decoder = ResNetBlock(bridge_channels, top_channels,    nl=self.nl, dims=self.dims, include_norm=False)

        self.encoders.append( encoder )
        self.bridges.append(  bridge )
        self.decoders.append( decoder )


# --- FLOW MATCHING WRAPPER AND TIME EMBEDDING ---

class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal time embedding module.
    Takes a 1D batch of times t (B,) and maps it to (B, dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Handle odd dimension
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1), "constant", 0)
            
        return embeddings


class FlowMatchingUNet(nn.Module):
    """
    Top-level wrapper for the Flow Matching model.
    
    This module:
    1. Takes (x_t, t, y_k) as input.
    2. Generates time embeddings for t.
    3. Concatenates (x_t, t_embed, y_k) along the channel dimension.
    4. Passes the combined tensor through the UNet.
    5. Returns the UNet's output (the predicted velocity).
    """
    def __init__(
        self,
        in_channels: int,           # Channels in x_t (e.g., 2 for vx, vz)
        condition_channels: int,    # Channels in y_k (e.g., 2 for vx, vz)
        time_embed_dim: int,        # Dimension for time embedding (e.g., 64)
        features: List[int] = (32, 64, 128) # UNet feature map sizes
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.condition_channels = condition_channels
        self.time_embed_dim = time_embed_dim

        # 1. Time embedding module
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # 2. Total channels for UNet input
        unet_in_channels = in_channels + time_embed_dim + condition_channels
        
        # 3. The UNet
        self.unet = UNet(
            in_channels=unet_in_channels,
            out_channels=in_channels, # Output is velocity, same channels as input
            features=features
        )
        
    def forward(self, 
                x_t: torch.Tensor,       # (B, C_in, H, W)
                t: torch.Tensor,         # (B,)
                y_k: torch.Tensor        # (B, C_cond, H, W)
               ) -> torch.Tensor:
        
        # 1. Get time embedding
        # t_embed: (B, time_embed_dim)
        t_embed = self.time_embed(t)
        
        # 2. Expand time embedding to spatial dimensions
        # (B, time_embed_dim) -> (B, time_embed_dim, 1, 1)
        t_embed_spatial = t_embed[:, :, None, None]
        # (B, time_embed_dim, 1, 1) -> (B, time_embed_dim, H, W)
        t_embed_expanded = t_embed_spatial.expand(-1, -1, x_t.shape[2], x_t.shape[3])
        
        # 3. Concatenate all inputs along the channel dimension
        model_input = torch.cat([x_t, t_embed_expanded, y_k], dim=1)
        
        # 4. Pass through UNet
        predicted_velocity = self.unet(model_input)
        
        return predicted_velocity