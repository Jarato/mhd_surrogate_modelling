# -*- coding: utf-8 -*-
# packages/flow_matching_2d/flow_matching_2d_base/src/flow_matching_2d_base/model.py
#
# Implements a conditional Flow Matching model for 2D data.
#
# --- MODIFICATION ---
# Integrated the physics-informed padding (PaddedConv2D) from the
# user's tcKAE model (study 5.2) into the UNet architecture.
# - The UNet's DoubleConv block now uses PaddedConv2D.
# - ResNetBlock is simplified, as padding is handled by PaddedConv2D.
# - Removed generic 'padding' arguments in favor of this specific implementation.
# --- END MODIFICATION ---

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# --- Utility (from original) ---
# This pad function is no longer used by ResNetBlock, but kept
# in case other modules need it. The new PaddedConv2D handles
# padding internally.
def pad(x: torch.Tensor, dim: int, extent: int, padding_mode: str) -> torch.Tensor:
    """
    Manually pads a specific dimension of a tensor.
    """
    if padding_mode == "zeros":
        padding_mode = "constant"
    
    num_dims = x.ndim
    pad_tuple = [0] * (2 * num_dims)
    pad_dim_index = (num_dims - 1) - dim 
    pad_tuple[2 * pad_dim_index] = extent
    pad_tuple[2 * pad_dim_index + 1] = extent

    return F.pad(x, tuple(pad_tuple), mode=padding_mode, value=0.0)

# --- Sinusoidal Time Embedding ---
# Standard module for embedding time 't' in flow/diffusion models.

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embeddings."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): A 1D tensor of time steps, shape (B,).
        
        Returns:
            torch.Tensor: Time embeddings, shape (B, dim).
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1: # Zero pad if dim is odd
             embeddings = F.pad(embeddings, (0, 1), "constant", 0)
        return embeddings

# --- NEW: Physics-Informed Padding (from tcKAE model) ---
class PaddedConv2D(nn.Module):
    """
    A custom 2D convolutional layer that applies physics-informed padding before
    the convolution operation. This is designed for the specific flow problem:
    - Z-axis (Top/Bottom): Zero-padding for no-slip walls.
    - Max X-axis (Right): Reflection-padding for the outlet.
    - Min X-axis (Left): Replication-padding for the inlet.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, # Default stride to 1, as DoubleConv uses
        padding: str = "same" # This argument is now ignored but kept for compatibility
    ):
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

# --- UNet Architecture (Ported from user-provided unet.py) ---

class MaxPool(nn.Module):
    def __init__(self, *ks):
        super().__init__()
        if len(ks) == 1:   self.mp = nn.MaxPool1d(ks, ks)
        elif len(ks) == 2: self.mp = nn.MaxPool2d(ks, ks)
        elif len(ks) == 3: self.mp = nn.MaxPool3d(ks, ks)
        else: raise Exception()

    def forward(self, x):
        return self.mp(x)

class DoubleConv(nn.Module):
    # --- MODIFIED ---
    # Now uses PaddedConv2D instead of nn.Conv2d
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 padding="same",      # Ignored, handled by PaddedConv2D
                 padding_mode="zeros",# Ignored, handled by PaddedConv2D
                 dims=2,
                 nl=nn.ReLU(),
                 include_norm=True):
        super().__init__()
        # Use our custom physics-padded conv layer
        conv_cls = PaddedConv2D 
        
        if dims != 2:
            raise ValueError("PaddedConv2D is only implemented for dims=2")

        self.norm = None
        if include_norm:
            norm_cls = [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d][dims-1]
            self.norm = norm_cls(num_features = out_channels, eps=1.0)

        self.nl = nl 
        # PaddedConv2D uses kernel_size=3, stride=1 (implied by "same" padding)
        self.conv1 = conv_cls(in_channels,  out_channels, kernel_size=3, stride=1)
        self.conv2 = conv_cls(out_channels, out_channels, kernel_size=3, stride=1)
        # --- END MODIFICATION ---

    def forward(self, x):
        x = self.conv1(x)
        if self.norm is not None: x = self.norm(x)
        x = self.nl(x)
        x = self.conv2(x)
        return x

class ResNetBlock(nn.Module):
    # --- MODIFIED ---
    # Simplified to use DoubleConv, which now handles its own padding.
    # Removed manual padding logic from the forward pass.
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 padding: Tuple[str, ...], # This is now IGNORED, but kept for UNet compatibility
                 dims=2,
                 nl=nn.ReLU(),
                 include_norm=True):
        super().__init__()
        conv_cls = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims-1]
        self.in_channels = in_channels
        self.out_channels = out_channels

        # DoubleConv now uses "valid" padding internally via PaddedConv2D
        # which applies the physics-informed padding *before* the conv.
        self.dc = DoubleConv(
            in_channels, out_channels, 
            padding="valid", # This is passed to DoubleConv
            dims=dims, nl=nl, include_norm=include_norm
        )
        self.conv = conv_cls(in_channels, out_channels, kernel_size=1)
        # self.padding = padding (No longer needed)
        self.dims = dims

    def forward(self, x):
        # x1, x2 = x.clone(), x.clone() (No longer need separate clones)
        
        # Manual padding is no longer needed here.
        # for pm, dim_idx in zip(self.padding, range(self.dims)):
        #     x1 = pad(x1, dim=2+dim_idx, extent=2, padding_mode=pm)

        y = self.conv(x) # 1x1 conv for residual
        z = self.dc(x)   # DoubleConv on *unpadded* input
        # --- END MODIFICATION ---

        return y + z
    
class UNet(nn.Module):
    # --- MODIFIED ---
    # The 'padding' argument is now passed down to ResNetBlock,
    # but ResNetBlock and DoubleConv will ignore it in favor
    # of the hard-coded physics-informed padding in PaddedConv2D.
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 features=(32, 64),
                 padding=("zeros", "zeros"), # Kept for API compatibility
                 nl=nn.ReLU()):
        super().__init__()
        self.padding = padding # Stored, but functionally ignored
        self.dims = len(self.padding)
        assert self.dims in [1,2,3]
        if self.dims != 2:
            raise ValueError("Physics-informed padding (PaddedConv2D) only supports dims=2")
            
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
        
        x_up = None # Initialize x_up
        for k in range(len(self.decoders)-1, -1, -1):
            if k == len(self.decoders)-1: 
                x_up = torch.zeros_like( x_bridged[k] )
            
            if x_up is None:
                raise RuntimeError("x_up should not be None in decoder loop")

            x = self.decoders[k]( x_bridged[k] + x_up )
            if k > 0:  x_up = self.up(x)
        return x

    def add_top_unet_block(self, in_channels, bridge_channels, out_channels):
        # Note: self.padding is passed, but will be ignored by ResNetBlock/DoubleConv
        encoder = ResNetBlock(in_channels,     bridge_channels, padding=self.padding, nl=self.nl, dims=self.dims, include_norm=False) 
        bridge  = ResNetBlock(bridge_channels, bridge_channels, padding=self.padding, nl=self.nl, dims=self.dims, include_norm=False) 
        decoder = ResNetBlock(bridge_channels, out_channels,    padding=self.padding, nl=self.nl, dims=self.dims, include_norm=False) 

        self.encoders.append( encoder )
        self.bridges.append(  bridge )
        self.decoders.append( decoder )
    
    def append_unet_block(self, bridge_channels):
        top_channels = self.encoders[-1].out_channels

        # Note: self.padding is passed, but will be ignored by ResNetBlock/DoubleConv
        encoder = ResNetBlock(top_channels,    bridge_channels, padding=self.padding, nl=self.nl, dims=self.dims, include_norm=False) 
        bridge  = ResNetBlock(bridge_channels, bridge_channels, padding=self.padding, nl=self.nl, dims=self.dims, include_norm=False) 
        decoder = ResNetBlock(bridge_channels, top_channels,    padding=self.padding, nl=self.nl, dims=self.dims, include_norm=False) 

        self.encoders.append( encoder )
        self.bridges.append(  bridge )
        self.decoders.append( decoder )


# --- Main Flow Matching Model ---

class FlowMatchingUNet(nn.Module):
    """
    Wrapper for the UNet to implement the conditional flow matching model.
    v_t_theta(x_t, t, y_k)
    """
    # --- MODIFIED ---
    # Removed 'unet_padding' argument as it's no longer used.
    def __init__(
        self,
        in_channels: int,           # Channels of x_t
        condition_channels: int,    # Channels of y_k
        time_embed_dim: int,
        features: List[int],        # UNet features, e.g., [64, 128, 256]
        # unet_padding: Tuple[str, ...] = ("zeros", "zeros") # <-- REMOVED
    ):
        super().__init__()
        
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        
        # Calculate total input channels for the UNet
        # We concatenate x_t, time_embedding, and condition y_k
        unet_in_channels = in_channels + time_embed_dim + condition_channels
        
        # The UNet output should match the channels of x_t,
        # as it's predicting the velocity (x_1 - x_0) which has
        # the same shape as x_t.
        unet_out_channels = in_channels 
        
        self.unet = UNet(
            in_channels=unet_in_channels,
            out_channels=unet_out_channels,
            features=features,
            # padding=unet_padding, # <-- REMOVED
            nl=nn.GELU() # Use GELU as in the tcKAE model
        )

    def forward(
        self, 
        x_t: torch.Tensor, # Interpolated state (B, C, H, W)
        t: torch.Tensor,   # Time (B,)
        y_k: torch.Tensor  # Condition (B, C, H, W)
    ) -> torch.Tensor:
        
        # 1. Get time embedding
        # t_embed: (B, time_embed_dim)
        t_embed = self.time_embedding(t)
        
        # 2. Expand time embedding to match spatial dims
        # t_embed_spatial: (B, time_embed_dim, H, W)
        B, C, H, W = x_t.shape
        t_embed_spatial = t_embed.view(B, -1, 1, 1).expand(B, -1, H, W)
        
        # 3. Concatenate all inputs along the channel dimension
        # model_input: (B, C + time_embed_dim + C_cond, H, W)
        model_input = torch.cat([x_t, t_embed_spatial, y_k], dim=1)
        
        # 4. Pass through UNet
        # predicted_velocity: (B, C, H, W)
        predicted_velocity = self.unet(model_input)
        
        return predicted_velocity

