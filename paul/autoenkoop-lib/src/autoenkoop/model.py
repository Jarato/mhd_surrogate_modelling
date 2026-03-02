from torch import nn


class ConvAutoencoderNorm(nn.Module):
    def __init__(self, latent_dim, intermediate_dim = 1024, use_bias = True):
        super().__init__()

        self.latent_dimension = latent_dim
        self.intermediate_dimension = intermediate_dim
        self.flattened_dim = 73728
        self.pre_2d_shape = (128, 72, 8)
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(self.flattened_dim, self.intermediate_dimension, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm1d(self.intermediate_dimension),
            nn.Linear(self.intermediate_dimension, self.latent_dimension, bias=use_bias),
        )

        self.linear_dynamics = nn.Linear(self.latent_dimension, self.latent_dimension, bias=False)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dimension, self.intermediate_dimension, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm1d(self.intermediate_dimension),
            nn.Linear(self.intermediate_dimension, self.flattened_dim, bias=use_bias),
            nn.GELU(), # REMOVE FOR OLD MODELS before 25.02.2026, 14:00
            nn.Unflatten(1, self.pre_2d_shape),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=0, bias=use_bias)
        )

    def forward(self, x_t):
        z_t = self.encoder(x_t)
        z_t_plus_1 = self.linear_dynamics(z_t)
        x_t_plus_1 = self.decoder(z_t_plus_1)
        return x_t_plus_1
    

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim, intermediate_dim = 1024, use_bias = True):
        super().__init__()

        self.latent_dimension = latent_dim
        self.intermediate_dimension = intermediate_dim
        self.flattened_dim = 73728
        self.pre_2d_shape = (128, 72, 8)
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(self.flattened_dim, self.intermediate_dimension, bias=use_bias),
            nn.GELU(),
            nn.Linear(self.intermediate_dimension, self.latent_dimension, bias=use_bias)
        )

        self.linear_dynamics = nn.Linear(self.latent_dimension, self.latent_dimension, bias=False)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dimension, self.intermediate_dimension, bias=use_bias),
            nn.GELU(),
            nn.Linear(self.intermediate_dimension, self.flattened_dim, bias=use_bias),
            nn.GELU(), # REMOVE FOR OLD MODELS before 25.02.2026, 14:00
            nn.Unflatten(1, self.pre_2d_shape),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=0, bias=use_bias)
        )

    def forward(self, x_t):
        z_t = self.encoder(x_t)
        z_t_plus_1 = self.linear_dynamics(z_t)
        x_t_plus_1 = self.decoder(z_t_plus_1)
        return x_t_plus_1
    
    
class ConvAutoencoderKoop(nn.Module):
    def __init__(self, latent_dim, intermediate_dim = 1024, use_bias = True):
        super().__init__()

        self.latent_dimension = latent_dim
        self.intermediate_dimension = intermediate_dim
        self.flattened_dim = 73728
        self.pre_2d_shape = (128, 72, 8)
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(self.flattened_dim, self.intermediate_dimension, bias=use_bias),
            nn.GELU(),
            nn.Linear(self.intermediate_dimension, self.latent_dimension, bias=use_bias),
        )

        #self.linear_dynamics = nn.Linear(self.latent_dimension, self.latent_dimension, bias=False)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dimension, self.intermediate_dimension, bias=use_bias),
            nn.GELU(),
            nn.Linear(self.intermediate_dimension, self.flattened_dim, bias=use_bias),
            nn.GELU(), # REMOVE FOR OLD MODELS before 25.02.2026, 14:00
            nn.Unflatten(1, self.pre_2d_shape),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.GELU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=0, bias=use_bias)
        )

    def forward(self, x_t):
        z_t = self.encoder(x_t)
        #z_t_plus_1 = self.linear_dynamics(z_t)
        x_t_recon = self.decoder(z_t)
        return z_t, x_t_recon