import torch
import torch.nn as nn
from tqdm import tqdm
from models.diffusion.ddpm import DDPM
from models.diffusion.forward import get_from_idx
from models.sparsityGuidance import sparsityGradient

# This is how we will use the model once trained
@torch.inference_mode()
def generate_ddpm(denoiser_model:nn.Module, backward_sampler:DDPM, cfg, device,verbose=True):
    # Set the model in evaluation mode
    denoiser_model.eval()
    # Noise from a normal distribution
    xnoisy = torch.randn((cfg.DIFFUSION.NSAMPLES, 4, cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS, cfg.DATASET.FUTURE_LEN), device=device)
    xnoisy_over_time = [xnoisy]
    # Now, to reverse the diffusion process, use a sequence of denoising steps
    for t in tqdm(iterable=reversed(range(0, backward_sampler.timesteps)),
                          dynamic_ncols=False,total=backward_sampler.timesteps,
                          desc="Sampling :: ", position=0):
        t_tensor = torch.as_tensor(t, dtype=torch.long, device=device).reshape(-1).expand(xnoisy.shape[0])
        # Estimate the noise
        eps_pred = denoiser_model(xnoisy, t_tensor)
        # Denoise with the sampler and the estimation of the noise
        xnoisy, sigma = backward_sampler.step(eps_pred, xnoisy, t)
        if cfg.DIFFUSION.GUIDANCE == "sparsity":
            # Update the noisy image with the sparsity guidance (TESTING!)
            sparsity_grad = sparsityGradient(xnoisy,cfg, device)
            xnoisy-= 0.004*sigma*sparsity_grad
        xnoisy_over_time.append(xnoisy)
    return xnoisy, xnoisy_over_time

@torch.inference_mode()
def generate_ddim(denoiser_model:nn.Module, taus, backward_sampler:DDPM, cfg, device, verbose=True):
    # Set the model in evaluation mode
    denoiser_model.eval()
    # Noise from a normal distribution
    xnoisy = torch.randn((cfg.DIFFUSION.NSAMPLES, 4, cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS, cfg.DATASET.FUTURE_LEN), device=device)
    last_t                     = torch.ones(xnoisy.shape[0], dtype=torch.long, device=device) * (backward_sampler.timesteps-1)
    sqrt_alpha_bar_t           = get_from_idx(backward_sampler.sqrt_alpha_bar, last_t)
    sqrt_one_minus_alpha_bar_t = get_from_idx(backward_sampler.sqrt_one_minus_alpha_bar, last_t)
    xnoisy_over_time = [xnoisy]
    # Now, to reverse the diffusion process, use a sequence of denoising steps
    for t in tqdm(iterable=reversed(taus),
                          dynamic_ncols=False,total=len(taus),
                          desc="Sampling :: ", position=0, disable=not verbose):
        # Time vectors
        ts = torch.ones(xnoisy.shape[0], dtype=torch.long, device=device) * t
        # Estimate the noise
        predicted_noise = denoiser_model(xnoisy, ts)
        # The betas, alphas etc.
        sqrt_alpha_bar_t_prev           = get_from_idx(backward_sampler.sqrt_alpha_bar, ts)
        sqrt_one_minus_alpha_bar_t_prev = get_from_idx(backward_sampler.sqrt_one_minus_alpha_bar, ts)
        # Predicted x0
        predicted_x0                    = (xnoisy-sqrt_one_minus_alpha_bar_t*predicted_noise)/sqrt_alpha_bar_t
        # Generating images for t-1 (deterministic way)
        xnoisy = sqrt_alpha_bar_t_prev * predicted_x0 + sqrt_one_minus_alpha_bar_t_prev * predicted_noise
        sqrt_alpha_bar_t                = sqrt_alpha_bar_t_prev
        sqrt_one_minus_alpha_bar_t      = sqrt_one_minus_alpha_bar_t_prev
        xnoisy_over_time.append(xnoisy)
    return xnoisy, xnoisy_over_time