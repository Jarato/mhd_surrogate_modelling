# Data preparation
- 70/30 train/test split
- temporal mean removed

# Model
- no decoder bias

A) Training Koopman matrix
- 128d latent state

B) DMD Koopman matrix
- 32d latent state

# Training
- 300 epochs
- first 10 epochs just reconstruction
- Loss Weighting (\lambda_{recon}, \lambda_{lin}):
    - [(1,1), (1, 10), (1,100)]
- lr_start = 1e-3
- lr scheduler


# Evaluation
- 100 time window prediction vs true MSE