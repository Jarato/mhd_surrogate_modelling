from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

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


class RelativeMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # Flatten everything except batch dimension
        batch_size = pred.shape[0]
        
        pred_flat = pred.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)

        numerator = torch.sum((pred_flat - target_flat) ** 2, dim=1)
        denominator = torch.sum(target_flat ** 2, dim=1).detach() + self.eps

        rel_mse = numerator / denominator  # per sample
        return rel_mse.mean()



class WSConv2d(nn.Conv2d):
    def __init__(self, *args, eps=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        
        fan_in = (
            self.in_channels
            * self.kernel_size[0]
            * self.kernel_size[1]
            / self.groups
        )

        scale = (1.0 / fan_in) ** 0.5

        self.gain = nn.Parameter(
            torch.ones(self.out_channels, 1, 1, 1) * scale
        )

    def forward(self, x):
        weight = self.weight

        # Mean over (in_channels, kernel_h, kernel_w)
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        weight = weight - mean

        # Std per output channel
        var = weight.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        weight = weight / torch.sqrt(var + self.eps)

        weight = weight * self.gain

        return F.conv2d(
            x, weight, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )
        
class WSConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, eps=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        
        fan_in = (
            self.in_channels
            * self.kernel_size[0]
            * self.kernel_size[1]
            / self.groups
        )

        scale = (1.0 / fan_in) ** 0.5

        self.gain = nn.Parameter(
            torch.ones(1, self.out_channels, 1, 1) * scale
        )

    def forward(self, x):
        weight = self.weight

        # Mean over (in_channels, kernel_h, kernel_w)
        mean = weight.mean(dim=(0, 2, 3), keepdim=True)
        weight = weight - mean

        var = weight.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        weight = weight / torch.sqrt(var + self.eps)

        weight = weight * self.gain

        return F.conv_transpose2d(
            x, weight, self.bias,
            self.stride, self.padding,
            self.output_padding,
            self.groups, self.dilation
        )
        
class WSLinear(nn.Linear):
    def __init__(self, *args, eps=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        
        fan_in = self.in_features
        scale = (1.0 / fan_in) ** 0.5

        self.gain = nn.Parameter(
            torch.ones(self.out_features, 1) * scale
        )

    def forward(self, x):
        weight = self.weight

        mean = weight.mean(dim=1, keepdim=True)
        weight = weight - mean

        var = weight.var(dim=1, unbiased=False, keepdim=True)
        weight = weight / torch.sqrt(var + self.eps)

        weight = weight * self.gain

        return F.linear(x, weight, self.bias)

class Scale(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
    
    def forward(self, x):
        return x*self.factor

def norm_selector(dimension, norm_type="none", bias=False):
    norm_dict = {
        "none" : nn.Identity(),
        "batch" : nn.BatchNorm1d(dimension, affine=bias),
        "rms" : nn.RMSNorm(dimension, elementwise_affine=bias), # will add a Scale afterwards, with factor 1/sqrt(dimension)
        "layer" : nn.RMSNorm(dimension, elementwise_affine=bias),
    }
    return norm_dict[norm_type]

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim, intermediate_dim = 1024, weight_standard=False, bias_terms=True, norm_type="batch", norm_latent="rms"): # "layer"
        super().__init__()

        self.latent_dimension = latent_dim
        self.intermediate_dimension = intermediate_dim
        self.flattened_dim = 73728
        self.pre_2d_shape = (128, 72, 8)
        # Encoder
        
        if not weight_standard and norm_type=="none":
            encoder_modules = [
                nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=bias_terms),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=bias_terms),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=bias_terms),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=bias_terms),
                nn.GELU(),
                nn.Flatten(),
                nn.Linear(self.flattened_dim, self.intermediate_dimension, bias=bias_terms),
                nn.GELU(),
                nn.Linear(self.intermediate_dimension, self.latent_dimension, bias=bias_terms)
            ]
            decoder_modules = [
                nn.Linear(self.latent_dimension, self.intermediate_dimension, bias=bias_terms),
                nn.GELU(),
                nn.Linear(self.intermediate_dimension, self.flattened_dim, bias=bias_terms),
                nn.GELU(),
                nn.Unflatten(1, self.pre_2d_shape),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias_terms),
                nn.GELU(),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias_terms),
                nn.GELU(),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias_terms),
                nn.GELU(),
                nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=0, bias=bias_terms)
            ]
        elif weight_standard:
            if norm_type=="standard":
                encoder_modules = [
                    WSConv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=bias_terms),
                    nn.GELU(),
                    WSConv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=bias_terms),
                    nn.GELU(),
                    WSConv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=bias_terms),
                    nn.GELU(),
                    WSConv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=bias_terms),
                    nn.GELU(),
                    nn.Flatten(),
                    WSLinear(self.flattened_dim, self.intermediate_dimension, bias=bias_terms),
                    nn.GELU(),
                    WSLinear(self.intermediate_dimension, self.latent_dimension, bias=bias_terms)
                ]
                decoder_modules = [
                    WSLinear(self.latent_dimension, self.intermediate_dimension, bias=bias_terms),
                    nn.GELU(),
                    WSLinear(self.intermediate_dimension, self.flattened_dim, bias=bias_terms),
                    nn.GELU(),
                    nn.Unflatten(1, self.pre_2d_shape),
                    WSConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias_terms),
                    nn.GELU(),
                    WSConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias_terms),
                    nn.GELU(),
                    WSConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias_terms),
                    nn.GELU(),
                    WSConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=0, bias=bias_terms)
                ]
            else:
                encoder_modules = [
                    weight_norm(nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=bias_terms)),
                    nn.GELU(),
                    weight_norm(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=bias_terms)),
                    nn.GELU(),
                    weight_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=bias_terms)),
                    nn.GELU(),
                    weight_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=bias_terms)),
                    nn.GELU(),
                    nn.Flatten(),
                    weight_norm(nn.Linear(self.flattened_dim, self.intermediate_dimension, bias=bias_terms)),
                    nn.GELU(),
                    weight_norm(nn.Linear(self.intermediate_dimension, self.latent_dimension, bias=bias_terms))
                ]
                decoder_modules = [
                    weight_norm(nn.Linear(self.latent_dimension, self.intermediate_dimension, bias=bias_terms)),
                    nn.GELU(),
                    weight_norm(nn.Linear(self.intermediate_dimension, self.flattened_dim, bias=bias_terms)),
                    nn.GELU(),
                    nn.Unflatten(1, self.pre_2d_shape),
                    weight_norm(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias_terms)),
                    nn.GELU(),
                    weight_norm(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias_terms)),
                    nn.GELU(),
                    weight_norm(nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias_terms)),
                    nn.GELU(),
                    weight_norm(nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=0, bias=bias_terms))
                ]
        else:
            encoder_modules = [
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
                nn.Linear(self.intermediate_dimension, self.latent_dimension, bias=bias_terms)
            ]
            encoder_modules.append(norm_selector(self.latent_dimension, norm_latent, bias=False))
            if norm_latent == "rms":
                factor = 1.0/np.sqrt(self.latent_dimension)
                encoder_modules.append(Scale(factor))
            decoder_modules = [
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
            ]
        
        self.encoder = nn.Sequential(*encoder_modules)
        self.decoder = nn.Sequential(*decoder_modules)
        self.linear_dynamics = nn.Linear(self.latent_dimension, self.latent_dimension, bias=False)


    def forward(self, x_t):
        z_t = self.encoder(x_t)
        z_t_plus_1 = self.linear_dynamics(z_t)
        x_t_plus_1 = self.decoder(z_t_plus_1)
        return x_t_plus_1



