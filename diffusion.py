import torch

def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class Diffusion:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise):
        sqrt_alpha_hat = self.alpha_hat[t]**0.5
        sqrt_one_minus = (1 - self.alpha_hat[t])**0.5
        return sqrt_alpha_hat * x0 + sqrt_one_minus * noise