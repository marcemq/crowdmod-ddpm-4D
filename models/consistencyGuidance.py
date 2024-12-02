import torch

def consistencyGradient(xnoisy: torch.Tensor, cfg, device, rho_min = 0.25) -> torch.Tensor:
    #compute the gradient of the consistency loss
    grad = torch.zeros(xnoisy.shape, device=device)
    mask = xnoisy[:, 0, :, :, :] < rho_min

    grad[:, 1, :, :, :] = torch.where(mask, grad[:, 1, :, :, :]*(1-grad[:, 0, :, :, :]/rho_min), torch.zeros_like(grad[:, 1, :, :, :]))
    grad[:, 2, :, :, :] = torch.where(mask, grad[:, 2, :, :, :]*(1-grad[:, 0, :, :, :]/rho_min), torch.zeros_like(grad[:, 2, :, :, :]))
    return grad  