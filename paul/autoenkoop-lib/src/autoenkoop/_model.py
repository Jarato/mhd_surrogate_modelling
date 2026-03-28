from torch import nn
import torch

class SchurLinear(nn.Module):
    def __init__(self, dimension, device):
        super().__init__()
        self.Q_param = torch.nn.Parameter(torch.rand((dimension,dimension), device=device))
        self.H_param = torch.nn.Parameter(torch.randn((dimension,dimension), device=device))
        self.weight = self.calc_weight()

    def calc_weight(self):
        Q = torch.linalg.qr(self.Q_param)[0]
        H = torch.triu(self.H_param, -1)
        self.weight = Q  @ H  @ Q.T
        return self.weight

    def forward(self, x):
        return nn.functional.linear(x, self.calc_weight())

class ConvAutoencoderSchur(nn.Module):
    def __init__(self, latent_dim, intermediate_dim = 1024, use_bias = True, device = 'cuda'):
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

        self.linear_dynamics = SchurLinear(self.latent_dimension, device)#nn.Linear(self.latent_dimension, self.latent_dimension, bias=False)

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

class ConvAutoencoderZeroDecoder(nn.Module):
    def __init__(self, latent_dim, intermediate_dim = 1024):
        super().__init__()

        self.latent_dimension = latent_dim
        self.intermediate_dimension = intermediate_dim
        self.flattened_dim = 73728
        self.pre_2d_shape = (128, 72, 8)
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(self.flattened_dim, self.intermediate_dimension, bias=True),
            nn.GELU(),
            nn.BatchNorm1d(self.intermediate_dimension),
            nn.Linear(self.intermediate_dimension, self.latent_dimension, bias=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dimension, self.intermediate_dimension, bias=False),
            nn.GELU(),
            nn.BatchNorm1d(self.intermediate_dimension),
            nn.Linear(self.intermediate_dimension, self.flattened_dim, bias=False),
            nn.GELU(), # REMOVE FOR OLD MODELS before 25.02.2026, 14:00
            nn.Unflatten(1, self.pre_2d_shape),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=0, bias=False)
        )

        self.linear_dynamics = nn.Linear(self.latent_dimension, self.latent_dimension, bias=False)


    def forward(self, x_t):
        z_t = self.encoder(x_t)
        z_t_plus_1 = self.linear_dynamics(z_t)
        x_t_plus_1 = self.decoder(z_t_plus_1)
        return x_t_plus_1

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim, intermediate_dim = 1024, bias_terms=True, norm_type="batch", norm_latent=True): # "layer"
        super().__init__()

        self.latent_dimension = latent_dim
        self.intermediate_dimension = intermediate_dim
        self.flattened_dim = 73728
        self.pre_2d_shape = (128, 72, 8)
        # Encoder
        
        if norm_latent:
            last_encoder_entry = nn.BatchNorm1d(self.latent_dimension, affine=bias_terms) if norm_type=="batch" else nn.RMSNorm(self.latent_dimension, elementwise_affine=bias_terms)
        else:
            last_encoder_entry = nn.Identity()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=bias_terms),
            nn.GELU(),
            nn.BatchNorm2d(16, affine=bias_terms) if norm_type=="batch" else nn.GroupNorm(1, 16, affine=bias_terms),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=bias_terms),
            nn.GELU(),
            nn.BatchNorm2d(32, affine=bias_terms) if norm_type=="batch" else nn.GroupNorm(1, 32, affine=bias_terms),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=bias_terms),
            nn.GELU(),
            nn.BatchNorm2d(64, affine=bias_terms) if norm_type=="batch" else nn.GroupNorm(1, 64, affine=bias_terms),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=bias_terms),
            nn.GELU(),
            nn.BatchNorm2d(128, affine=bias_terms) if norm_type=="batch" else nn.GroupNorm(1, 128, affine=bias_terms),
            nn.Flatten(),
            nn.Linear(self.flattened_dim, self.intermediate_dimension, bias=bias_terms),
            nn.GELU(),
            nn.BatchNorm1d(self.intermediate_dimension, affine=bias_terms) if norm_type=="batch" else nn.LayerNorm(self.intermediate_dimension, elementwise_affine=bias_terms),
            nn.Linear(self.intermediate_dimension, self.latent_dimension, bias=bias_terms),
            last_encoder_entry#nn.BatchNorm1d(self.latent_dimension, affine=bias_terms) if norm_type=="batch" else nn.RMSNorm(self.latent_dimension, elementwise_affine=bias_terms)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dimension, self.intermediate_dimension, bias=bias_terms),
            nn.GELU(),
            nn.BatchNorm1d(self.intermediate_dimension, affine=bias_terms) if norm_type=="batch" else nn.LayerNorm(self.intermediate_dimension, elementwise_affine=bias_terms),
            nn.Linear(self.intermediate_dimension, self.flattened_dim, bias=bias_terms),
            nn.GELU(),
            nn.Unflatten(1, self.pre_2d_shape),
            nn.BatchNorm2d(128, affine=bias_terms) if norm_type=="batch" else nn.GroupNorm(1, 128, affine=bias_terms),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias_terms),
            nn.GELU(),
            nn.BatchNorm2d(64, affine=bias_terms) if norm_type=="batch" else nn.GroupNorm(1, 64, affine=bias_terms),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias_terms),
            nn.GELU(),
            nn.BatchNorm2d(32, affine=bias_terms) if norm_type=="batch" else nn.GroupNorm(1, 32, affine=bias_terms),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias_terms),
            nn.GELU(),
            nn.BatchNorm2d(16, affine=bias_terms) if norm_type=="batch" else nn.GroupNorm(1, 16, affine=bias_terms),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=0, bias=bias_terms)
        )

        self.linear_dynamics = nn.Linear(self.latent_dimension, self.latent_dimension, bias=False)


    def forward(self, x_t):
        z_t = self.encoder(x_t)
        z_t_plus_1 = self.linear_dynamics(z_t)
        x_t_plus_1 = self.decoder(z_t_plus_1)
        return x_t_plus_1



