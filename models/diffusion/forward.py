import torch
import torch.nn as nn

def get_from_idx(element: torch.Tensor, idx: torch.Tensor):
    ele = element.gather(-1, idx)
    return ele.reshape(-1, 1, 1, 1)

# This class implements the forward diffusion process
class ForwardSampler(nn.Module):
    def __init__(self, timesteps=1000, scale=1, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        # Total number of steps in the diffusion process
        self.timesteps = timesteps
        # The betas and the alphas
        beta = torch.linspace(
                scale*beta_start,
                scale*beta_end,
                self.timesteps,
                dtype=torch.float32
            )
        self.register_buffer("beta", beta)
        # Some intermediate values we will need
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(self.alpha_bar))
        self.register_buffer("one_by_sqrt_alpha", 1. / torch.sqrt(self.alpha))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1 - self.alpha_bar))

    def forward(self, x0: torch.Tensor, timesteps: torch.Tensor):
        # Generate normal noise
        epsilon = torch.randn_like(x0)
        # Get the mean/standard deviation for the queried timesteps
        mean    = get_from_idx(self.sqrt_alpha_bar, timesteps) * x0      # Mean
        std_dev = get_from_idx(self.sqrt_one_minus_alpha_bar, timesteps) # Standard deviation
        # Sample is mean plus the scaled noise
        sample  = mean + std_dev * epsilon
        return sample, epsilon