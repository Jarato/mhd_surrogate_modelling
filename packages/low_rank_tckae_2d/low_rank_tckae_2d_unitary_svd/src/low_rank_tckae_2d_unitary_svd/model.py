# -*- coding: utf-8 -*-
# packages/mhd_q2d_tckae/src/mhd_q2d_tckae/model_svd.py
# Note: This is a modified version for 2D data (X, Z spatial dims)
# with a new SVD-based Koopman operator.
#
# --- MODIFICATION ---
# The model now dynamically supports three modes via flags:
# 1. (use_flattened_for_koopman=True):
#    Koopman op on the high-dimensional flattened feature space (F).
# 2. (use_flattened_for_koopman=False, use_bottleneck=True):
#    Koopman op on a lower-dim latent space (d), with an intermediate
#    bottleneck layer (b). Path: F -> b -> d
# 3. (use_flattened_for_koopman=False, use_bottleneck=False):
#    Koopman op on a lower-dim latent space (d), with a direct
#    FC layer. Path: F -> d

from collections import OrderedDict
from typing import Any, Dict, List
import logging # <-- Added for logging info

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CUSTOM MODULE FOR PHYSICS-INFORMED PADDING ---
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


class Encoder2D(nn.Module):
    """
    A true 2D CNN Encoder, now using the custom PaddedConv2D layers.
    --- MODIFIED ---
    The fully-connected network is now just a Flatten operation.
    The output is the high-dimensional flattened feature map.
    """
    def __init__(
        self,
        in_channels: int,
        # latent_dim: int, <-- REMOVED
    ):
        super().__init__()
        # self.latent_dim = latent_dim <-- REMOVED
        self.in_channels = in_channels

        # --- MODIFIED: Replaced nn.Conv2d with our new PaddedConv2D ---
        self.conv_network = nn.Sequential(
            PaddedConv2D(self.in_channels, 32, kernel_size=3, stride=2),
            nn.GELU(),
            nn.BatchNorm2d(32),
            PaddedConv2D(32, 64, kernel_size=3, stride=2),
            nn.GELU(),
            nn.BatchNorm2d(64),
            PaddedConv2D(64, 128, kernel_size=3, stride=2),
            nn.GELU(),
            nn.BatchNorm2d(128),
            PaddedConv2D(128, 256, kernel_size=3, stride=2),
            # nn.GELU(), <-- REMOVED this line as it was redundant
            nn.BatchNorm2d(256),
        )
        
        self.fc_network = nn.Sequential(
            nn.Flatten(),
            # --- MODIFIED ---
            # The Linear layers are no longer added here.
            # The output of the encoder is the flattened feature map.
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Input x is expected to be (B, C, X, Z)
        x = self.conv_network(x)
        x = self.fc_network(x)
        return x


class Decoder2D(nn.Module):
    """
    A true 2D CNN Decoder.
    --- MODIFIED ---
    This module is now just the "un-convolution". It takes the
    high-dimensional flattened feature map as input.
    """
    def __init__(
        self,
        # latent_dim: int, <-- REMOVED
        # bottleneck_dim: int, <-- REMOVED
        out_channels: int,
        # encoder_flattened_size: int, <-- REMOVED
        conv_output_shape: tuple[int, ...],
        target_spatial_dims: tuple[int, int],
        # use_bottleneck: bool = True, <-- REMOVED
    ):
        super().__init__()
        # self.latent_dim = latent_dim <-- REMOVED
        self.out_channels = out_channels
        # self.encoder_flattened_size = encoder_flattened_size <-- REMOVED
        self.conv_output_shape = conv_output_shape
        self.target_spatial_dims = target_spatial_dims

        # --- MODIFIED ---
        # The fc_network has been removed.
        # --- END MODIFICATION ---

        self.conv_transpose_network = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # <-- Fixed typo
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        # --- MODIFIED ---
        # x = self.fc_network(z) <-- REMOVED
        # The input 'z' is now the flattened feature map.
        # We just need to reshape it.
        x = z.view(-1, *self.conv_output_shape)
        # --- END MODIFICATION ---
        x = self.conv_transpose_network(x)
        
        # Crop to target dimension if necessary (due to conv arithmetic)
        s_x, s_z = self.target_spatial_dims
        x = x[:, :, :s_x, :s_z]
        
        return x


class tcKoopmanAutoencoder2D(nn.Module):
    """
    The main Temporally-Consistent 2D Koopman Autoencoder model.
    
    --- MODIFIED ---
    The dynamics can now operate on either the flattened feature space 'F'
    or a bottlenecked latent space 'd', controlled by flags.
    
    'latent_dim' (d) is the size of the Koopman space (either F or d_bottleneck)
    'koopman_rank' (r) is the rank of the low-rank approximation.
    """
    def __init__(
        self,
        in_channels: int,
        koopman_rank: int, # <-- 'r' of the SVD operator
        input_spatial_dims: tuple[int, int],
        steps: int,
        steps_back: int,
        steps_tc: int,
        sequence_length: int, # M
        # --- MODIFIED DYNAMIC ARGUMENTS ---
        use_flattened_for_koopman: bool = True,
        use_bottleneck: bool = True, # <-- RE-INTRODUCED
        latent_dim: int | None = None,      # 'd' (if not using flattened)
        bottleneck_dim: int | None = None,  # 'b' (if not using flattened and using bottleneck)
    ):
        super().__init__()
        self.steps = steps
        self.steps_back = steps_back
        self.steps_tc = steps_tc
        self.koopman_rank = koopman_rank
        self.use_flattened_for_koopman = use_flattened_for_koopman
        self.use_bottleneck = use_bottleneck # <-- NEW
        
        x_dim, z_dim = input_spatial_dims
        
        # Encoder is now just the ConvNet + Flatten
        self.encoder = Encoder2D(in_channels)
        
        # We still need to do a dummy pass to find the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, x_dim, z_dim)
            conv_output_before_flatten = self.encoder.conv_network(dummy_input)
            flattened_output = self.encoder.fc_network(conv_output_before_flatten)
            
            flattened_size = flattened_output.shape[1]      # This is 'F'
            conv_output_shape = conv_output_before_flatten.shape[1:]

        # Decoder is now just the ConvTransposeNet
        self.decoder = Decoder2D(
            out_channels=in_channels,
            conv_output_shape=conv_output_shape,
            target_spatial_dims=input_spatial_dims,
        )

        # --- MODIFIED: CONDITIONAL BOTTLENECK AND KOOPMAN SPACE ---
        
        # Default: Koopman space is the flattened space
        self.bottleneck_to_latent = nn.Identity()
        self.latent_to_bottleneck = nn.Identity()
        
        if self.use_flattened_for_koopman:
            # STATE 1: Koopman op on flattened space (F)
            self.koopman_space_dim = flattened_size # 'd' = 'F'
            logging.info(
                f"Using flattened space for Koopman. Operator space d={self.koopman_space_dim}"
            )
        else:
            # STATE 2 or 3: Koopman op on latent space (d)
            if latent_dim is None:
                raise ValueError(
                    "If 'use_flattened_for_koopman' is False, 'latent_dim' must be provided."
                )
            
            self.koopman_space_dim = latent_dim # 'd' = 'd_bottleneck'

            if self.use_bottleneck:
                # STATE 2: Flattened -> Bottleneck -> Latent
                if bottleneck_dim is None:
                    raise ValueError(
                        "If 'use_bottleneck' is True (and not using flattened), "
                        "'bottleneck_dim' must be provided."
                    )
                
                logging.info(
                    f"Using bottleneck: Flattened ({flattened_size}) -> "
                    f"Bottleneck ({bottleneck_dim}) -> Latent ({latent_dim})"
                )
                
                self.bottleneck_to_latent = nn.Sequential(
                    nn.Linear(flattened_size, bottleneck_dim),
                    nn.GELU(),
                    nn.Linear(bottleneck_dim, latent_dim)
                )
                self.latent_to_bottleneck = nn.Sequential(
                    nn.Linear(latent_dim, bottleneck_dim),
                    nn.GELU(),
                    nn.Linear(bottleneck_dim, flattened_size)
                )
            
            else:
                # STATE 3: Flattened -> Latent (No Bottleneck)
                logging.info(
                    f"Using direct connection (no bottleneck): "
                    f"Flattened ({flattened_size}) -> Latent ({latent_dim})"
                )
                self.bottleneck_to_latent = nn.Sequential(
                    nn.Linear(flattened_size, latent_dim)
                )
                self.latent_to_bottleneck = nn.Sequential(
                    nn.Linear(latent_dim, flattened_size)
                )
        
        # --- NEW: Final activation before Koopman space ---
        self.final_encoder_activation = nn.GELU()
        # --- END NEW ---

        # self.latent_dim is a convenience attribute for the *actual* size
        # of the space where the Koopman operator lives.
        self.latent_dim = self.koopman_space_dim
        logging.info(
            f"Initializing Koopman SVD operator in space d={self.latent_dim} "
            f"with rank r={self.koopman_rank}"
        )
        
        # --- END OF CONDITIONAL BLOCK ---


        # --- SVD Koopman Operator Parameters ---
        # K = U * Sigma * V^T
        # self.koopman_space_dim is the 'd' (either F or d_bottleneck)
        # self.koopman_rank is 'r'
        # U_learn: (d, r)
        # V_learn: (d, r)
        # Sigma_learn: (r,)
        self.U_learn = nn.Parameter(torch.empty(self.koopman_space_dim, self.koopman_rank))
        self.V_learn = nn.Parameter(torch.empty(self.koopman_space_dim, self.koopman_rank))
        self.Sigma_learn = nn.Parameter(torch.empty(self.koopman_rank))
        
        # Initialize parameters for stability
        nn.init.orthogonal_(self.U_learn.data)
        nn.init.orthogonal_(self.V_learn.data)
        nn.init.ones_(self.Sigma_learn.data)
        # --- END OF SVD MODIFICATION ---


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input image 'x' into the Koopman space.
        Applies a final GELU activation before the Koopman space.
        """
        z_flat = self.encoder(x)
        z_before_activation = self.bottleneck_to_latent(z_flat)
        z_koopman = self.final_encoder_activation(z_before_activation)
        return z_koopman

    def decode(self, z_koopman: torch.Tensor) -> torch.Tensor:
        """
        Decodes a state 'z_koopman' from the Koopman space back to the image space.
        If using bottleneck, z_latent -> z_flat -> x_recon
        If not,            z_flat -> (Identity) -> z_flat -> x_recon
        """
        z_flat = self.latent_to_bottleneck(z_koopman)
        x_recon = self.decoder(z_flat)
        return x_recon

    def koopman_step(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies the forward Koopman operator: z_k+1 = z_k * K
        where K = U * Sigma * V^T
        """
        # z: (B, d) where d is self.koopman_space_dim
        # U_learn: (d, r)
        # Sigma_learn: (r,)
        # V_learn: (d, r)
        
        # z_u = z @ U -> (B, r)
        z_u = z @ self.U_learn
        
        # z_u_sigma = (z @ U) * Sigma -> (B, r)
        # We multiply element-wise, broadcasting Sigma
        z_u_sigma = z_u * self.Sigma_learn
        
        # z_k = (z @ U * Sigma) @ V^T -> (B, d)
        z_k = z_u_sigma @ self.V_learn.t()
        
        return z_k

    def koopman_step_backward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies the backward Koopman operator: z_k-1 = z_k * K^-1
        where K^-1 = V * Sigma^-1 * U^T
        """
        # z: (B, d) where d is self.koopman_space_dim
        
        # z_v = z @ V -> (B, r)
        z_v = z @ self.V_learn
        
        # Calculate Sigma^-1 with stability epsilon
        sigma_inv = 1.0 / (self.Sigma_learn + 1e-8)
        
        # z_v_sigma_inv = (z @ V) * Sigma^-1 -> (B, r)
        z_v_sigma_inv = z_v * sigma_inv
        
        # z_k = (z @ V * Sigma^-1) @ U^T -> (B, d)
        z_k = z_v_sigma_inv @ self.U_learn.t()
        
        return z_k

    def forward(
        self,
        x: torch.Tensor,
        mode: str = 'forward',
    ) -> Dict[str, List[torch.Tensor]]:
        if mode == 'forward':
            max_steps = max(self.steps, self.steps_tc)
            op = self.koopman_step
            state_key = "predicted_states"
            latent_key = "latent_states"
        elif mode == 'backward':
            max_steps = self.steps_back
            op = self.koopman_step_backward
            state_key = "predicted_states_back"
            latent_key = "latent_states_back"
        else:
            raise ValueError(f"Unknown mode: {mode}")
        # z is now the state in the correct Koopman space (flattened or bottlenecked)
        z = self.encode(x)
        
        predicted_states = []
        latent_states = []
        
        z_k = z
        for _ in range(max_steps):
            z_k = op(z_k)
            latent_states.append(z_k)
            # self.decode() correctly handles decoding from either space
            predicted_states.append(self.decode(z_k))
            
        # self.decode() correctly handles decoding from either space
        predicted_states.append(self.decode(z))
        
        return {state_key: predicted_states, latent_key: latent_states}

