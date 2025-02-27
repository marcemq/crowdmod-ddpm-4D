import torch

def sparsityGradient(xnoisy: torch.Tensor, cfg, device) -> torch.Tensor:
    grad = torch.zeros(xnoisy.shape, device=device)
    grad[:,0,:,:]=torch.sign(xnoisy[:,0,:,:])
    # Compute the gradient of the sparsity loss
    return grad

def preservationMass(x: torch.Tensor, device, delta_t=1.0, delta_l=1.0) -> torch.Tensor:
    """
    Guidance for mass preservation within the defined grid
    Return:
    - grad of shame shape as xnoisy
    """
    B, C, H, W, L = x.shape
    energy = torch.zeros(x.shape, device=device)
    # temporal term for rho eq. 6
    energy[:, 0, :, :, :-1] += (1 / delta_t) * (x[:, 0, :, :, 1:] - x[:, 0, :, :, :-1])
    # spatial term for vel eq. 7
    energy[:, 0, :-1, :, :] += (1 / delta_l) * x[:, 0, :-1, :, :] * (x[:, 1, :-1, :, :] - x[:, 1, :, :, :] + x[:, 2, :-1, :, :] - x[:, 2, :, :, :])
    # spatial term for vel eq. 8
    energy[:, 0, :, :-1, :] += (1 / delta_l) * x[:, 0, :, :-1, :] * (x[:, 1, :, :-1, :] - x[:, 1, :, :, :])
    # spatial term for vel eq. 9
    energy[:, 0, :, :-1, :] += (1 / delta_l) * x[:, 0, :, :-1, :] * (x[:, 2, :, :-1, :] - x[:, 0, :, :-1, :])
    # derivative with respect to x (i,j,t)
    grad = 0
    return grad