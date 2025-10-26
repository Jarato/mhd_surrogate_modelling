# -*- coding: utf-8 -*-
# packages/mhd_q2d_tckae_uncons_ab/src/mhd_q2d_tckae_uncons_ab/model.py
# Note: This is a modified version for 2D data (X, Z spatial dims)
# with an unconstrained low-rank AB-Decomposition Koopman operator.
#
# --- MODIFICATION ---
# The Koopman operator K is now parameterized as K = A @ B,
# where A is (d, r) and B is (r, d).
# The backward operator K_inv is parameterized as C @ D.
# All SVD-related constraints are removed.

from collections import OrderedDict
from typing import Any, Dict, List
import logging
import math # <-- Added for initialization

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
    The fully-connected network is now just a Flatten operation.
    The output is the high-dimensional flattened feature map.
    """
    def __init__(
        self,
        in_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels

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
    This module is now just the "un-convolution". It takes the
    high-dimensional flattened feature map as input.
    """
    def __init__(
        self,
        out_channels: int,
        conv_output_shape: tuple[int, ...],
        target_spatial_dims: tuple[int, int],
    ):
        super().__init__()
        self.out_channels = out_channels
        self.conv_output_shape = conv_output_shape
        self.target_spatial_dims = target_spatial_dims

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
        # The input 'z' is now the flattened feature map.
        # We just need to reshape it.
        x = z.view(-1, *self.conv_output_shape)
        x = self.conv_transpose_network(x)
        
        # Crop to target dimension if necessary (due to conv arithmetic)
        s_x, s_z = self.target_spatial_dims
        x = x[:, :, :s_x, :s_z]
        
        return x


class tcKoopmanAutoencoder2D(nn.Module):
    """
    The main Temporally-Consistent 2D Koopman Autoencoder model.
    
    --- MODIFIED (AB Decomposition) ---
    The dynamics are defined by K = A @ B and K_inv = C @ D,
    where A, B, C, D are learnable unconstrained matrices.
    
    'latent_dim' (d) is the size of the Koopman space (either F or d_bottleneck)
    'koopman_rank' (r) is the rank of the low-rank approximation.
    """
    def __init__(
        self,
        in_channels: int,
        koopman_rank: int, # <-- 'r' of the AB operator
        input_spatial_dims: tuple[int, int],
        steps: int,
        steps_back: int,
        steps_tc: int,
        sequence_length: int, # M
        use_flattened_for_koopman: bool = True,
        use_bottleneck: bool = True,
        latent_dim: int | None = None,      # 'd' (if not using flattened)
        bottleneck_dim: int | None = None,  # 'b' (if not using flattened and using bottleneck)
    ):
        super().__init__()
        self.steps = steps
        self.steps_back = steps_back
        self.steps_tc = steps_tc
        self.koopman_rank = koopman_rank
        self.use_flattened_for_koopman = use_flattened_for_koopman
        self.use_bottleneck = use_bottleneck
        
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

        # --- CONDITIONAL BOTTLENECK AND KOOPMAN SPACE ---
        
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
        
        # --- Final activation before Koopman space ---
        self.final_encoder_activation = nn.GELU()

        # self.latent_dim is a convenience attribute for the *actual* size
        # of the space where the Koopman operator lives.
        self.latent_dim = self.koopman_space_dim
        logging.info(
            f"Initializing Koopman AB operator in space d={self.latent_dim} "
            f"with rank r={self.koopman_rank}"
        )
        
        # --- END OF CONDITIONAL BLOCK ---


        # --- AB Koopman Operator Parameters ---
        # K = A * B
        # self.koopman_space_dim is 'd'
        # self.koopman_rank is 'r'
        
        # A_learn: (d, r)
        self.A_learn = nn.Parameter(torch.empty(self.koopman_space_dim, self.koopman_rank))
        # B_learn: (r, d)
        self.B_learn = nn.Parameter(torch.empty(self.koopman_rank, self.koopman_space_dim))
        
        # Backward operator K_inv = C * D
        # C_learn: (d, r)
        self.C_learn = nn.Parameter(torch.empty(self.koopman_space_dim, self.koopman_rank))
        # D_learn: (r, d)
        self.D_learn = nn.Parameter(torch.empty(self.koopman_rank, self.koopman_space_dim))

        # Initialize parameters
        self._init_weights()
        # --- END OF AB MODIFICATION ---


    def _init_weights(self):
        """Initializes the AB/CD matrices using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.A_learn.data, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B_learn.data, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.C_learn.data, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.D_learn.data, a=math.sqrt(5))


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
        """
        z_flat = self.latent_to_bottleneck(z_koopman)
        x_recon = self.decoder(z_flat)
        return x_recon

    def koopman_step(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies the forward Koopman operator: z_k+1 = z_k * K
        where K = A * B
        """
        # z: (B, d)
        # A_learn: (d, r)
        # B_learn: (r, d)
        
        # z_a = z @ A -> (B, r)
        z_a = z @ self.A_learn
        
        # z_k = (z @ A) @ B -> (B, d)
        z_k = z_a @ self.B_learn
        
        return z_k

    def koopman_step_backward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies the backward Koopman operator: z_k-1 = z_k * K_inv
        where K_inv = C * D (learned)
        """
        # z: (B, d)
        # C_learn: (d, r)
        # D_learn: (r, d)

        # z_c = z @ C -> (B, r)
        z_c = z @ self.C_learn
        
        # z_k = (z @ C) @ D -> (B, d)
        z_k = z_c @ self.D_learn
        
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
            predicted_states.append(self.decode(z_k))
            
        predicted_states.append(self.decode(z))
        
        return {state_key: predicted_states, latent_key: latent_states}
