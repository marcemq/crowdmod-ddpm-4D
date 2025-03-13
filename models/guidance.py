import torch
import logging

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

def preservationMassNumericalGradient(x, device, delta_t=0.5, delta_l=1.0, eps=0.01) -> torch.Tensor:
    B, C, H, W, L = x.shape
    N = C * H * W * L  # Total elements per sample
    grad_energy = torch.zeros_like(x)  # Gradient tensor

    # Compute E(x) once
    E_x = compute_energy(x, delta_t, delta_l)  # Shape: (B,)
    logging.info(f'Value range of batched E_x {E_x}')
    # Flatten spatial dimensions for efficient perturbation indexing
    x_flat = x.view(B, N)  # Shape: (B, N)

    # Generate perturbation mask: Identity matrix for each sample
    perturb_mask = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

    # Create perturbed versions of x
    x_perturbed = x_flat.unsqueeze(1) + eps * perturb_mask  # Shape: (B, N, N)

    # Reshape back to (B, C, H, W, L) for compute_energy
    x_perturbed = x_perturbed.view(B * N, C, H, W, L)

    # Compute E(x + eps) in parallel
    E_x_perturbed = compute_energy(x_perturbed, delta_t, delta_l)  # Shape: (B * N,)
    E_x_perturbed = E_x_perturbed.view(B, N)  # Reshape back to (B, N)

    # Compute finite difference gradient
    grad_flat = (E_x_perturbed - E_x.unsqueeze(1)) / eps  # Shape: (B, N)

    # Reshape back to (B, C, H, W, L)
    grad_energy = grad_flat.view(B, C, H, W, L)

    return grad_energy

def preservationMassNumericalGradient_base(x, device, delta_t=0.5, delta_l=1.0, eps=0.01) -> torch.Tensor:
    B, C, H, W, L = x.shape
    grad_energy = torch.zeros_like(x)  # Store gradients

    # Compute E(x)
    E_x = compute_energy(x, delta_t, delta_l)
    logging.info(f'Value range of batched E_x {E_x}')

    # Loop through each voxel
    for b in range(B):
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    for t in range(L):
                        # Perturb x[b, c, i, j, t] by eps
                        x_perturbed = x.clone()
                        x_perturbed[b, c, i, j, t] += eps

                        # Compute E(x + eps)
                        E_x_perturbed = compute_energy(x_perturbed, delta_t, delta_l)

                        # Compute finite difference approximation of gradient
                        grad_energy[b, c, i, j, t] = (E_x_perturbed[b] - E_x[b]) / eps

    return grad_energy