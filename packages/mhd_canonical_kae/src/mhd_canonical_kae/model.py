# -*- coding: utf-8 -*-
# mhd_canonical_kae/src/mhd_canonical_kae/model.py

from collections import OrderedDict

import torch
import torch.nn as nn

# Note: The specific architecture (channels, kernel sizes, etc.) is a
# starting point and will likely need to be tuned based on the specific
# characteristics of the MHD data and the desired latent space dimension.


class Encoder(nn.Module):
    """
    A 3D CNN Encoder to compress spatio-temporal data into a latent vector.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.network = nn.Sequential(
            # Input: (B, C_in, D, H, W) e.g., (B, 20, 577, 97, 97)
            # For simplicity in this example, we assume smaller, more manageable dimensions
            # after some initial downsampling or for a test case.
            # Let's assume input is (B, 20, 64, 64, 64) for this example architecture.
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 32, 32, 32, 32)
            nn.GELU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 16, 16, 16)
            nn.GELU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 8, 8, 8)
            nn.GELU(),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # -> (B, 256, 4, 4, 4)
            nn.GELU(),
            nn.BatchNorm3d(256),
            nn.Flatten(),  # -> (B, 256 * 4 * 4 * 4) = (B, 16384)
            nn.Linear(256 * 4 * 4 * 4, self.latent_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


class Decoder(nn.Module):
    """
    A 3D CNN Decoder to reconstruct spatio-temporal data from a latent vector.
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.linear = nn.Linear(self.latent_dim, 256 * 4 * 4 * 4)

        self.network = nn.Sequential(
            # Input: (B, 256, 4, 4, 4)
            nn.ConvTranspose3d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (B, 128, 8, 8, 8)
            nn.GELU(),
            nn.BatchNorm3d(128),
            nn.ConvTranspose3d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (B, 64, 16, 16, 16)
            nn.GELU(),
            nn.BatchNorm3d(64),
            nn.ConvTranspose3d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (B, 32, 32, 32, 32)
            nn.GELU(),
            nn.BatchNorm3d(32),
            nn.ConvTranspose3d(
                32, self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (B, C_out, 64, 64, 64)
            # No final activation, as the output can be any real value.
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        # Reshape the latent vector back into a 3D feature map
        x = self.linear(z)
        x = x.view(-1, 256, 4, 4, 4)
        return self.network(x)


class KoopmanAutoencoder(nn.Module):
    """
    The main Koopman Autoencoder model.

    This class combines the Encoder, Decoder, and a linear Koopman operator
    to learn the dynamics of a system.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

        # The Koopman operator is a single linear layer.
        # It approximates the evolution of the system in the latent space.
        self.koopman_operator = nn.Linear(latent_dim, latent_dim, bias=False)

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes the input data into the latent space."""
        return self.encoder(x)

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes a latent vector back to the original data space."""
        return self.decoder(z)

    def koopman_step(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Applies the Koopman operator to advance the latent state."""
        return self.koopman_operator(z)

    def forward(
        self,
        x_t: torch.Tensor,
        x_t_plus_1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Performs a full forward pass, returning all tensors needed for the loss.

        Args:
            x_t (torch.Tensor): The batch of system states at time t.
            x_t_plus_1 (torch.Tensor): The batch of system states at time t+1.

        Returns:
            A dictionary containing all the necessary output tensors:
            - 'z_t': Latent representation of x_t.
            - 'z_t_plus_1_encoded': Latent representation of x_t_plus_1.
            - 'z_t_plus_1_predicted': Latent state at t+1 predicted from z_t.
            - 'x_t_reconstructed': Reconstruction of the input x_t.
            - 'x_t_plus_1_predicted': Prediction of the next state x_t_plus_1.
        """
        # Encode both current and future states
        z_t = self.encode(x_t)
        z_t_plus_1_encoded = self.encode(x_t_plus_1)

        # Predict the future latent state using the Koopman operator
        z_t_plus_1_predicted = self.koopman_step(z_t)

        # Reconstruct the original input state from its latent representation
        x_t_reconstructed = self.decode(z_t)

        # Predict the future state by decoding the predicted future latent state
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


# This block allows you to test the script directly
if __name__ == "__main__":
    # --- How to use the KoopmanAutoencoder ---

    # Define model parameters
    # Note: The input shape for this test is smaller than the real data
    # to match the example architecture defined above.
    batch_size = 4
    in_channels = 20
    latent_dim = 128
    d, h, w = 64, 64, 64  # Dummy spatial dimensions

    # 1. Instantiate the model
    model = KoopmanAutoencoder(
        in_channels=in_channels,
        latent_dim=latent_dim,
    )

    # 2. Create dummy input tensors
    # The data needs to be in (B, C, D, H, W) format for 3D CNNs
    dummy_x_t = torch.randn(batch_size, in_channels, d, h, w)
    dummy_x_t_plus_1 = torch.randn(batch_size, in_channels, d, h, w)

    # 3. Perform a forward pass
    outputs = model(
        dummy_x_t,
        dummy_x_t_plus_1,
    )

    # 4. Check the shapes of the outputs
    print("--- Model Test ---")
    print(f"Input shape: {dummy_x_t.shape}")
    for name, tensor in outputs.items():
        print(f"Output '{name}' shape: {tensor.shape}")

    # --- Verification ---
    # Latent vectors should have shape (batch_size, latent_dim)
    assert outputs["z_t"].shape == (batch_size, latent_dim)
    # Reconstructed/predicted data should have the same shape as the input
    assert outputs["x_t_reconstructed"].shape == dummy_x_t.shape
    assert outputs["x_t_plus_1_predicted"].shape == dummy_x_t.shape
    print("\nModel test passed successfully!")
