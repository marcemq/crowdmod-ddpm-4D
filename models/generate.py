import torch
import torch.nn as nn
from tqdm import tqdm
from models.diffusion.ddpm import DDPM
from models.diffusion.forward import get_from_idx
from models.guidance import sparsityGradient, preservationMassNumericalGradientOptimal

# This is how we will use the model once trained
@torch.inference_mode()
def generate_ddpm(denoiser_model:nn.Module, past:torch.Tensor, backward_sampler:DDPM, cfg, device, nsamples, history=False, mprops_count=4):
    # Set the model in evaluation mode
    denoiser_model.eval()
    # Noise from a normal distribution
    xnoisy = torch.randn((nsamples, mprops_count, cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS, cfg.DATASET.FUTURE_LEN), device=device)
    xnoisy_over_time = [xnoisy]
    # Now, to reverse the diffusion process, use a sequence of denoising steps
    for t in tqdm(iterable=reversed(range(0, backward_sampler.timesteps)),
                          dynamic_ncols=False,total=backward_sampler.timesteps,
                          desc="DDPM Sampling :: ", position=0):
        t_tensor = torch.as_tensor(t, dtype=torch.long, device=device).reshape(-1).expand(xnoisy.shape[0])
        # Estimate the noise
        eps_pred = denoiser_model(xnoisy, t_tensor, past)
        # Denoise with the sampler and the estimation of the noise
        xnoisy, sigma, alpha_t = backward_sampler.step(eps_pred, xnoisy, t)
        if cfg.MODEL.DDPM.GUIDANCE == "sparsity":
            # Update the noisy image with the sparsity guidance
            sparsity_grad = sparsityGradient(xnoisy, cfg, device)
            xnoisy-= 0.004*sigma*sparsity_grad # 0.004*sqrt(1-alpha_t)
        if cfg.MODEL.DDPM.GUIDANCE == "mass_preservation":
            mass_preserv_grad = preservationMassNumericalGradientOptimal(xnoisy, device, delta_t=1.0, delta_l=1.0, eps=0.1)
            xnoisy-= (1-alpha_t)*mass_preserv_grad
        if history:
            xnoisy_over_time.append(xnoisy)

    if not history:
        xnoisy_over_time.append(xnoisy)

    return xnoisy, xnoisy_over_time

@torch.inference_mode()
def generate_ddim(denoiser_model:nn.Module, past:torch.Tensor, taus, backward_sampler:DDPM, cfg, device, nsamples, history=False, mprops_count=4):
    # Set the model in evaluation mode
    denoiser_model.eval()
    # Noise from a normal distribution
    xnoisy = torch.randn((nsamples, mprops_count, cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS, cfg.DATASET.FUTURE_LEN), device=device)
    last_t                     = torch.ones(xnoisy.shape[0], dtype=torch.long, device=device) * (backward_sampler.timesteps-1)
    beta_t                     = get_from_idx(backward_sampler.beta, last_t)
    sqrt_alpha_bar_t           = get_from_idx(backward_sampler.sqrt_alpha_bar, last_t)
    sqrt_one_minus_alpha_bar_t = get_from_idx(backward_sampler.sqrt_one_minus_alpha_bar, last_t)
    xnoisy_over_time = [xnoisy]
    # Now, to reverse the diffusion process, use a sequence of denoising steps
    for t in tqdm(iterable=reversed(taus),
                          dynamic_ncols=False,total=len(taus),
                          desc="DDIM Sampling :: ", position=0):
        # Time vectors
        ts = torch.ones(xnoisy.shape[0], dtype=torch.long, device=device) * t
        # Estimate the noise
        predicted_noise = denoiser_model(xnoisy, ts, past)
        # The betas, alphas etc.
        beta_t_prev                     = get_from_idx(backward_sampler.beta, ts)
        sqrt_alpha_bar_t_prev           = get_from_idx(backward_sampler.sqrt_alpha_bar, ts)
        sqrt_one_minus_alpha_bar_t_prev = get_from_idx(backward_sampler.sqrt_one_minus_alpha_bar, ts)
        # Predicted x0
        predicted_x0                    = (xnoisy-sqrt_one_minus_alpha_bar_t*predicted_noise)/sqrt_alpha_bar_t
        # AR: Generating images for t-1 (deterministic way). Review this step, can we do it no deterministic?
        # AR: redo eq 65, 67 that depends on sigma and test, with sigma=0, and sigma=1
        xnoisy = sqrt_alpha_bar_t_prev * predicted_x0 + sqrt_one_minus_alpha_bar_t_prev * predicted_noise
        if cfg.MODEL.DDPM.GUIDANCE == "sparsity":
            # Update the noisy image with the sparsity guidance
            sparsity_grad = sparsityGradient(xnoisy, cfg, device)
            sigma = torch.sqrt(beta_t)
            xnoisy-= 0.004*sigma*sparsity_grad

        beta_t                          = beta_t_prev
        sqrt_alpha_bar_t                = sqrt_alpha_bar_t_prev
        sqrt_one_minus_alpha_bar_t      = sqrt_one_minus_alpha_bar_t_prev
        if history:
            xnoisy_over_time.append(xnoisy)

    if not history:
        xnoisy_over_time.append(xnoisy)

    return xnoisy, xnoisy_over_time

@torch.inference_mode()
def generate_fm(unet_model:nn.Module, past:torch.Tensor, cfg, device, nsamples, history=False, mprops_count=4):
    # Set the model in evaluation mode
    unet_model.eval()
    # Noise from a normal distribution
    xt = torch.randn((nsamples, mprops_count, cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS, cfg.DATASET.FUTURE_LEN), device=device)
    time_max_pos=cfg.MODEL.FLOW_MATCHING.TIME_MAX_POS
    delta = 1 / cfg.MODEL.FLOW_MATCHING.EULER_STEPS

    pbar = tqdm(range(1, cfg.MODEL.FLOW_MATCHING.EULER_STEPS + 1), desc="Sampling")

    # Cycle over the integration steps
    for i, t in enumerate(torch.linspace(0, 1, cfg.MODEL.FLOW_MATCHING.EULER_STEPS, device=device), start=1):
      time_indices = (t * time_max_pos).clamp(0, time_max_pos-1).long()
      time_indices = time_indices.to(device).expand(xt.size(0))
      # Apply the velocity to get the velocity
      u = unet_model(xt, time_indices, past)
      # Integration step
      xt           = xt + delta * u
      pbar.update(1)

    pbar.close()
    return xt

@torch.inference_mode()
def generate_convGRU(convGRU_model, x_test, y_test, teacher_forcing):
    # Set the model in evaluation mode
    convGRU_model.eval()
    predictions = convGRU_model(x_test, y_test, teacher_forcing)
    #AR: check if exp() is still needed for rho, vx and vy
    predictions[:,0,:,:,:] = torch.exp(predictions[:,0,:,:,:])
    predictions[:,3,:,:,:] = torch.exp(predictions[:,3,:,:,:])

    return predictions