# -*- coding: utf-8 -*-
# packages/mhd_q2d_tckae/src/mhd_q2d_tckae/model_2d.py
# Note: This is a modified version for 2D data (X, Z spatial dims)
# with custom physics-informed padding.

from collections import OrderedDict
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- NEW CUSTOM MODULE FOR PHYSICS-INFORMED PADDING ---
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
    """
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim
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
            nn.GELU(),
            nn.BatchNorm2d(256),
        )
        
        self.fc_network = nn.Sequential(
            nn.Flatten(),
            # The Linear layers will be added dynamically.
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
    (No changes needed here as padding is handled by the encoder).
    """
    def __init__(
        self,
        latent_dim: int,
        bottleneck_dim: int,
        out_channels: int,
        encoder_flattened_size: int,
        conv_output_shape: tuple[int, ...],
        target_spatial_dims: tuple[int, int],
        use_bottleneck: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.encoder_flattened_size = encoder_flattened_size
        self.conv_output_shape = conv_output_shape
        self.target_spatial_dims = target_spatial_dims

        fc_layers = []
        if use_bottleneck:
            fc_layers.extend([
                nn.Linear(self.latent_dim, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, self.encoder_flattened_size),
            ])
        else:
            fc_layers.append(
                nn.Linear(self.latent_dim, self.encoder_flattened_size)
            )
        self.fc_network = nn.Sequential(*fc_layers)

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
            nn.ConvTranspose2d(32, self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        x = self.fc_network(z)
        x = x.view(-1, *self.conv_output_shape)
        x = self.conv_transpose_network(x)
        
        # Crop to target dimension if necessary (due to conv arithmetic)
        s_x, s_z = self.target_spatial_dims
        x = x[:, :, :s_x, :s_z]
        
        return x


class tcKoopmanAutoencoder2D(nn.Module):
    """The main Temporally-Consistent 2D Koopman Autoencoder model."""
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        input_spatial_dims: tuple[int, int],
        steps: int,
        steps_back: int,
        steps_tc: int,
        sequence_length: int, # M
        bottleneck_dim: int = 4096,
        use_bottleneck: bool = True,
    ):
        super().__init__()
        self.use_bottleneck = use_bottleneck
        self.steps = steps
        self.steps_back = steps_back
        self.steps_tc = steps_tc
        self.latent_dim = latent_dim
        
        x_dim, z_dim = input_spatial_dims
        
        self.encoder = Encoder2D(in_channels, latent_dim)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, x_dim, z_dim)
            
            conv_output = self.encoder.conv_network(dummy_input)
            flattened_size = conv_output.flatten(1).shape[1]
            
            if self.use_bottleneck:
                self.encoder.fc_network.add_module(
                    "1", nn.Linear(flattened_size, bottleneck_dim)
                )
                self.encoder.fc_network.add_module("2", nn.GELU())
                self.encoder.fc_network.add_module(
                    "3", nn.Linear(bottleneck_dim, latent_dim)
                )
            else:
                self.encoder.fc_network.add_module(
                    "1", nn.Linear(flattened_size, latent_dim)
                )

            conv_output_shape = conv_output.shape[1:]

        self.decoder = Decoder2D(
            latent_dim=latent_dim,
            bottleneck_dim=bottleneck_dim,
            out_channels=in_channels,
            encoder_flattened_size=flattened_size,
            conv_output_shape=conv_output_shape,
            target_spatial_dims=input_spatial_dims,
            use_bottleneck=self.use_bottleneck,
        )

        self.koopman_operator = nn.Linear(latent_dim, latent_dim, bias=False)
        self.koopman_operator_backward = nn.Linear(latent_dim, latent_dim, bias=False)
        
        with torch.no_grad():
            self.koopman_operator_backward.weight.data = torch.pinverse(
                self.koopman_operator.weight.data.t()
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def koopman_step(self, z: torch.Tensor) -> torch.Tensor:
        return self.koopman_operator(z)

    def koopman_step_backward(self, z: torch.Tensor) -> torch.Tensor:
        return self.koopman_operator_backward(z)

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
