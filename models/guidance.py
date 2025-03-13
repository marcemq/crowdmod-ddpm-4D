import torch

def sparsityGradient(xnoisy: torch.Tensor, cfg, device) -> torch.Tensor:
    grad = torch.zeros(xnoisy.shape, device=device)
    grad[:,0,:,:]=torch.sign(xnoisy[:,0,:,:])
    # Compute the gradient of the sparsity loss
    return grad

def compute_energy(x: torch.Tensor, delta_t=0.5, delta_l=1.0) -> torch.Tensor:
    """
    Compute the energy function E_c(x) using vectorized operations.
    
    Args:
        x: Tensor of shape (B, 3, H, W, L)
        delta_t: Time step difference
        delta_l: Spatial step difference
    
    Returns:
        energy: Tensor of shape (B,) representing the computed energy for each batch element.
    """
    B, _, H, W, L = x.shape
    
    # Compute finite differences
    term1 = (1 / delta_t) * (x[:, 0, 1:-1, 1:-1, 1:] - x[:, 0, 1:-1, 1:-1, :-1])  # Temporal diff
    
    term2 = (1 / delta_l) * x[:, 0, 1:-1, 1:-1, :-1] * ((x[:, 1, 2:, 1:-1, :-1] - x[:, 1, 1:-1, 1:-1, :-1]) + (x[:, 2, 1:-1, 2:, :-1] - x[:, 2, 1:-1, 1:-1, :-1]))  # Mixed spatial terms

    term3 = (1 / delta_l) * (x[:, 0, 2:, 1:-1, :-1] - x[:, 0, 1:-1, 1:-1, :-1]) * x[:, 1, 1:-1, 1:-1, :-1]  # Spatial x-diff
    
    term4 = (1 / delta_l) * (x[:, 0, 1:-1, 2:, :-1] - x[:, 0, 1:-1, 1:-1, :-1]) * x[:, 2, 1:-1, 1:-1, :-1]  # Spatial y-diff
    
    # Compute f_x
    f_x = term1 + term2 + term3 + term4  # Expected shape: (B, H-2, W-2, L-1)

    # Ensure f_x has 5 dimensions (B, 1, H-2, W-2, L-1)
    f_x = f_x.unsqueeze(1)  # Adds channel dimension (C=1)
    # Compute energy by summing over spatial and temporal dimensions
    energy = 0.5 * torch.sum(f_x ** 2, dim=(1, 2, 3, 4))  # Shape: (B,)

    return energy