# -*- coding: utf-8 -*-
# packages/mhd_q2d_kae/src/mhd_q2d_kae/model.py

from collections import OrderedDict

import torch
import torch.nn as nn


class EncoderQ2D(nn.Module):
    """
    A Quasi-2D CNN Encoder. It treats the Y-dimension as part of the channel
    information and performs 2D convolutions on the X-Z plane.
    """

    def __init__(
        self,
        in_channels: int,
        y_dim: int,
        latent_dim: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.y_dim = y_dim

        # The effective number of channels for the 2D convolution
        effective_channels = self.in_channels * self.y_dim

        self.conv_network = nn.Sequential(
            nn.Conv2d(effective_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(256),
        )
        
        self.fc_network = nn.Sequential(
            nn.Flatten(),
            # The Linear layer will be added dynamically.
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Input shape: (B, C, X, Y, Z)
        B, C, X, Y, Z = x.shape
        
        # Reshape to (B, C*Y, X, Z) for 2D convolution
        x = x.permute(0, 1, 3, 2, 4).reshape(B, C * Y, X, Z)
        
        x = self.conv_network(x)
        x = self.fc_network(x)
        return x


class DecoderQ2D(nn.Module):
    """
    A Quasi-2D CNN Decoder, symmetric to the Encoder.
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        y_dim: int,
        encoder_flattened_size: int,
        conv_output_shape: tuple[int, ...],
        target_spatial_dims: tuple[int, int, int],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.y_dim = y_dim
        self.encoder_flattened_size = encoder_flattened_size
        self.conv_output_shape = conv_output_shape
        self.target_spatial_dims = target_spatial_dims

        self.fc_network = nn.Linear(self.latent_dim, self.encoder_flattened_size)

        self.conv_transpose_network = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, self.out_channels * self.y_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        x = self.fc_network(z)
        x = x.view(-1, *self.conv_output_shape)
        x = self.conv_transpose_network(x)
        
        # Reshape back to 5D tensor: (B, C*Y, X, Z) -> (B, C, Y, X, Z) -> (B, C, X, Y, Z)
        B, _, X, Z = x.shape
        x = x.view(B, self.out_channels, self.y_dim, X, Z)
        x = x.permute(0, 1, 3, 2, 4)
        
        # Crop the output to the target size internally.
        s_x, s_y, s_z = self.target_spatial_dims
        x = x[:, :, :s_x, :s_y, :s_z]
        
        return x


class KoopmanAutoencoderQ2D(nn.Module):
    """
    The main Quasi-2D Koopman Autoencoder model.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        input_spatial_dims: tuple[int, int, int],
        **kwargs,
    ):
        super().__init__()
        x_dim, y_dim, z_dim = input_spatial_dims
        
        self.encoder = EncoderQ2D(in_channels, y_dim, latent_dim)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, x_dim, y_dim, z_dim)
            B, C, X, Y, Z = dummy_input.shape
            dummy_reshaped = dummy_input.permute(0, 1, 3, 2, 4).reshape(B, C * Y, X, Z)
            
            conv_output = self.encoder.conv_network(dummy_reshaped)
            flattened_size = conv_output.flatten(1).shape[1]
            self.encoder.fc_network.add_module("1", nn.Linear(flattened_size, latent_dim))
            conv_output_shape = conv_output.shape[1:]

        self.decoder = DecoderQ2D(
            latent_dim=latent_dim,
            out_channels=in_channels,
            y_dim=y_dim,
            encoder_flattened_size=flattened_size,
            conv_output_shape=conv_output_shape,
            target_spatial_dims=input_spatial_dims,
        )

        self.koopman_operator = nn.Linear(latent_dim, latent_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def koopman_step(self, z: torch.Tensor) -> torch.Tensor:
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
