# -*- coding: utf-8 -*-
# packages/mhd_q2d_kae/src/mhd_q2d_kae/model.py

from collections import OrderedDict

import torch
import torch.nn as nn

# Your EncoderQ2D and DecoderQ2D classes are excellent and can be used directly.
# I've copied them here for completeness within the new model file.

class EncoderQ2D(nn.Module):
    """
    A Quasi-2D CNN Encoder with a progressive linear bottleneck.
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

        effective_channels = self.in_channels * self.y_dim

        self.conv_network = nn.Sequential(
            nn.Conv2d(effective_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
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
            # The Linear layers will be added dynamically.
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, C, X, Y, Z = x.shape
        x = x.permute(0, 1, 3, 2, 4).reshape(B, C * Y, X, Z)
        x = self.conv_network(x)
        x = self.fc_network(x)
        return x


class DecoderQ2D(nn.Module):
    """
    A Quasi-2D CNN Decoder with an optional progressive linear bottleneck.
    """

    def __init__(
        self,
        latent_dim: int,
        bottleneck_dim: int,
        out_channels: int,
        y_dim: int,
        encoder_flattened_size: int,
        conv_output_shape: tuple[int, ...],
        target_spatial_dims: tuple[int, int, int],
        use_bottleneck: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.y_dim = y_dim
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
            nn.ConvTranspose2d(32, self.out_channels * self.y_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        x = self.fc_network(z)
        x = x.view(-1, *self.conv_output_shape)
        x = self.conv_transpose_network(x)
        
        B, _, X, Z = x.shape
        x = x.view(B, self.out_channels, self.y_dim, X, Z)
        x = x.permute(0, 1, 3, 2, 4)
        
        s_x, s_y, s_z = self.target_spatial_dims
        x = x[:, :, :s_x, :s_y, :s_z]
        
        return x


class tcKoopmanAutoencoderQ2D(nn.Module):
    """
    The main Temporally-Consistent Quasi-2D Koopman Autoencoder model.
    This model integrates the tcKAE logic with the Q2D convolutional architecture.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        input_spatial_dims: tuple[int, int, int],
        steps: int,
        steps_back: int,
        steps_tc: int,
        bottleneck_dim: int = 4096,
        use_bottleneck: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.use_bottleneck = use_bottleneck
        self.steps = steps
        self.steps_back = steps_back
        self.steps_tc = steps_tc
        
        x_dim, y_dim, z_dim = input_spatial_dims
        
        self.encoder = EncoderQ2D(in_channels, y_dim, latent_dim)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, x_dim, y_dim, z_dim)
            B, C, X, Y, Z = dummy_input.shape
            dummy_reshaped = dummy_input.permute(0, 1, 3, 2, 4).reshape(B, C * Y, X, Z)
            
            conv_output = self.encoder.conv_network(dummy_reshaped)
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

        self.decoder = DecoderQ2D(
            latent_dim=latent_dim,
            bottleneck_dim=bottleneck_dim,
            out_channels=in_channels,
            y_dim=y_dim,
            encoder_flattened_size=flattened_size,
            conv_output_shape=conv_output_shape,
            target_spatial_dims=input_spatial_dims,
            use_bottleneck=self.use_bottleneck,
        )

        self.koopman_operator = nn.Linear(latent_dim, latent_dim, bias=False)
        
        # Backward operator for consistency loss, as in cKAE/tcKAE paper's implementation
        self.koopman_operator_backward = nn.Linear(latent_dim, latent_dim, bias=False)
        self.koopman_operator_backward.weight.data = torch.pinverse(self.koopman_operator.weight.data.T)

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
    ) -> dict[str, list[torch.Tensor]]:
        """
        Performs the forward pass, generating multi-step predictions.
        
        Args:
            x: Input tensor (batch of initial states).
            mode: 'forward' or 'backward' to control prediction direction.

        Returns:
            A dictionary containing lists of predicted states and latent states.
        """
        z = self.encode(x)
        q = z.clone() # Use clone to avoid in-place modification issues

        predicted_states = []
        latent_states = []

        if mode == 'forward':
            # Determine the maximum number of steps to iterate
            max_steps = max(self.steps, self.steps_tc)
            for _ in range(max_steps):
                q = self.koopman_step(q)
                predicted_states.append(self.decode(q))
                latent_states.append(q)
            
            # Also include the reconstruction of the original input
            predicted_states.append(self.decode(z))
            latent_states.append(z)

            return {
                "predicted_states": predicted_states,
                "latent_states": latent_states
            }

        elif mode == 'backward':
            for _ in range(self.steps_back):
                q = self.koopman_step_backward(q)
                predicted_states.append(self.decode(q))
                latent_states.append(q)

            predicted_states.append(self.decode(z))
            latent_states.append(z)
            
            return {
                "predicted_states_back": predicted_states,
                "latent_states_back": latent_states,
            }
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
