# -*- coding: utf-8 -*-
# mhd_canonical_kae/src/mhd_canonical_kae/model.py

from collections import OrderedDict

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    A 3D CNN Encoder that dynamically adapts to input spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv_network = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm3d(256),
        )
        
        self.fc_network = nn.Sequential(
            nn.Flatten(),
            # The Linear layer will be added dynamically.
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv_network(x)
        x = self.fc_network(x)
        return x


class Decoder(nn.Module):
    """
    A 3D CNN Decoder that dynamically adapts and crops its output.
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        encoder_flattened_size: int,
        conv_output_shape: tuple[int, ...],
        target_spatial_dims: tuple[int, int, int],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.encoder_flattened_size = encoder_flattened_size
        self.conv_output_shape = conv_output_shape
        self.target_spatial_dims = target_spatial_dims

        self.fc_network = nn.Linear(self.latent_dim, self.encoder_flattened_size)

        self.conv_transpose_network = nn.Sequential(
            nn.ConvTranspose3d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.GELU(),
            nn.BatchNorm3d(128),
            nn.ConvTranspose3d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.GELU(),
            nn.BatchNorm3d(64),
            nn.ConvTranspose3d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.GELU(),
            nn.BatchNorm3d(32),
            nn.ConvTranspose3d(
                32, self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            # Add a Tanh activation to ensure output is in [-1, 1]
            nn.Tanh(),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        x = self.fc_network(z)
        x = x.view(-1, *self.conv_output_shape)
        x = self.conv_transpose_network(x)
        
        # Crop the output to the target size internally.
        s_d, s_h, s_w = self.target_spatial_dims
        x = x[:, :, :s_d, :s_h, :s_w]
        
        return x


class KoopmanAutoencoder(nn.Module):
    """
    The main Koopman Autoencoder model.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        input_spatial_dims: tuple[int, int, int],
        **kwargs, # <-- THE FIX IS HERE
    ):
        """
        Initializes the model.
        **kwargs is used to accept and ignore extra config parameters
        like 'channels_used' that are not needed for model construction
        but are useful for checkpointing.
        """
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *input_spatial_dims)
            conv_output = self.encoder.conv_network(dummy_input)
            
            flattened_size = conv_output.flatten(1).shape[1]
            self.encoder.fc_network.add_module(
                "1", nn.Linear(flattened_size, latent_dim)
            )
            conv_output_shape = conv_output.shape[1:]

        self.decoder = Decoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            encoder_flattened_size=flattened_size,
            conv_output_shape=conv_output_shape,
            target_spatial_dims=input_spatial_dims,
        )

        self.koopman_operator = nn.Linear(latent_dim, latent_dim, bias=False)

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(x)

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(z)

    def koopman_step(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return self.koopman_operator(z)

    def forward(
        self,
        x_t: torch.Tensor,
        x_t_plus_1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_t = self.encode(x_t)
        z_t_plus_1_encoded = self.encode(x_t_plus_1)
        z_t_plus_1_predicted = self.koopman_step(z_t)
        x_t_reconstructed = self.decode(z_t)
        x_t_plus_1_predicted = self.decode(z_t_plus_1_predicted)

        return OrderedDict(
            [
                ("z_t", z_t),
                ("z_t_plus_1_encoded", z_t_plus_1_encoded),
                ("z_t_plus_1_predicted", z_t_plus_1_predicted),
                ("x_t_reconstructed", x_t_reconstructed),
                ("x_t_plus_1_predicted", x_t_plus_1_predicted),
            ]
        )
