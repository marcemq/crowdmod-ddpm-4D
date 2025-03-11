import torch

def sparsityGradient(xnoisy: torch.Tensor, cfg, device) -> torch.Tensor:
    grad = torch.zeros(xnoisy.shape, device=device)
    grad[:,0,:,:]=torch.sign(xnoisy[:,0,:,:])
    # Compute the gradient of the sparsity loss
    return grad

import torch

def compute_energy_slice(x: torch.Tensor, delta_t=0.5, delta_l=1.0) -> torch.Tensor:
    """
    Compute the energy function E_c(x) for the given tensor x.
    
    Args:
        x: Tensor of shape (B, 3, H, W, L)
        delta_t: Time step difference
        delta_l: Spatial step difference
    
    Returns:
        Scalar tensor representing the total energy.
    """
    B, _, H, W, L = x.shape
    
    # Define inner grid range to avoid out-of-bounds access
    i_range = slice(1, H - 1)
    j_range = slice(1, W - 1)
    t_range = slice(1, L - 1)
    
    # Compute terms inside the sum
    term1 = (1 / delta_t) * (x[:, 0, i_range, j_range, 1:] - x[:, 0, i_range, j_range, :-1])
    
    term2 = (1 / delta_l) * x[:, 0, i_range, j_range, :-1] * (
        x[:, 1, i_range + 1, j_range, :-1] - x[:, 1, i_range, j_range, :-1] +
        x[:, 2, i_range, j_range + 1, :-1] - x[:, 2, i_range, j_range, :-1]
    )
    
    term3 = (1 / delta_l) * (x[:, 0, i_range + 1, j_range, :-1] - x[:, 0, i_range, j_range, :-1]) * x[:, 1, i_range, j_range, :-1]
    
    term4 = (1 / delta_l) * (x[:, 0, i_range, j_range + 1, :-1] - x[:, 0, i_range, j_range, :-1]) * x[:, 2, i_range, j_range, :-1]
    
    # Compute energy sum
    energy = 0.5 * torch.sum((term1 + term2 + term3 + term4) ** 2)
    
    return energy

import torch

def compute_energy_base(x: torch.Tensor, delta_t=0.5, delta_l=1.0) -> torch.Tensor:
    """
    Compute the energy function E_c(x) using explicit loops.
    
    Args:
        x: Tensor of shape (B, 3, H, W, L)
        delta_t: Time step difference
        delta_l: Spatial step difference
    
    Returns:
        energy: Scalar tensor representing the computed energy
    """
    B, _, H, W, L = x.shape
    energy = torch.zeros(B, device=x.device)
    
    for b in range(B):
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                for t in range(L - 1):
                    term1 = (1 / delta_t) * (x[b, 0, i, j, t + 1] - x[b, 0, i, j, t])
                    term2 = (1 / delta_l) * x[b, 0, i, j, t] * (x[b, 1, i + 1, j, t] - x[b, 1, i, j, t] + x[b, 2, i, j + 1, t] - x[b, 2, i, j, t])
                    term3 = (1 / delta_l) * (x[b, 0, i + 1, j, t] - x[b, 0, i, j, t]) * x[b, 1, i, j, t]
                    term4 = (1 / delta_l) * (x[b, 0, i, j + 1, t] - x[b, 0, i, j, t]) * x[b, 2, i, j, t]
                    
                    f_x = term1 + term2 + term3 + term4
                    energy[b] += 0.5 * f_x ** 2
    
    return energy


import torch

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
    
    term2 = (1 / delta_l) * x[:, 0, 1:-1, 1:-1, :-1] * (
        (x[:, 1, 2:, 1:-1, :-1] - x[:, 1, 1:-1, 1:-1, :-1]) + 
        (x[:, 2, 1:-1, 2:, :-1] - x[:, 2, 1:-1, 1:-1, :-1])
    )  # Mixed spatial terms

    term3 = (1 / delta_l) * (x[:, 0, 2:, 1:-1, :-1] - x[:, 0, 1:-1, 1:-1, :-1]) * x[:, 1, 1:-1, 1:-1, :-1]  # Spatial x-diff
    
    term4 = (1 / delta_l) * (x[:, 0, 1:-1, 2:, :-1] - x[:, 0, 1:-1, 1:-1, :-1]) * x[:, 2, 1:-1, 1:-1, :-1]  # Spatial y-diff
    
    # Compute f_x
    f_x = term1 + term2 + term3 + term4

    # Compute energy by summing over spatial and temporal dimensions
    energy = 0.5 * torch.sum(f_x ** 2, dim=(1, 2, 3, 4))  # Shape: (B,)

    return energy

def preservationMassNumericalGradient(x, device, delta_t=0.5, delta_l=1.0, eps=0.01) -> torch.Tensor:
    B, C, H, W, L = x.shape
    N = C * H * W * L  # Total elements per sample
    grad_energy = torch.zeros_like(x)  # Gradient tensor

    # Compute E(x) once
    E_x = compute_energy(x, delta_t, delta_l)  # Shape: (B,)
    print(f'value range of E_x given batch of macroprops seqs')
    print(E_x)
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
    print(f'value range of E_x given batch of macroprops seqs')
    print(E_x)

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

def preservationMassGradient_cero(x, device, delta_t=0.5, delta_l=1.0) -> torch.Tensor:
    """
    Compute the gradient of the energy function defined for mass preservation.

    Args:
        x: Tensor of shape (B, 3, H, W, L)
        delta_t: Time step difference
        delta_l: Spatial step difference

    Returns:
        grad_E: Tensor of the same shape as x, representing the energy gradient
    """
    B, _, H, W, L = x.shape
    grad_E = torch.zeros_like(x, device=device)

    # Define the inner grid range
    i_range = torch.arange(1, H - 1, device=device)
    j_range = torch.arange(1, W - 1, device=device)

    # First term: temporal finite difference
    f1 = (1 / delta_t) * (x[:, 0, i_range, j_range, 1:] - x[:, 0, i_range, j_range, :-1])

    # Second term: spatial interaction term
    f2 = (1 / delta_l) * x[:, 0, i_range, j_range, :] * (x[:, 1, i_range + 1, j_range, :] - x[:, 1, i_range, j_range, :] +
          x[:, 2, i_range, j_range + 1, :] - x[:, 2, i_range, j_range, :])

    # Third term: x[1] interaction
    f3 = (1 / delta_l) * ((x[:, 0, i_range + 1, j_range, :] - x[:, 0, i_range, j_range, :]) * x[:, 1, i_range, j_range, :])

    # Fourth term: x[2] interaction
    f4 = (1 / delta_l) * ((x[:, 0, i_range, j_range + 1, :] - x[:, 0, i_range, j_range, :]) * x[:, 2, i_range, j_range, :])

    # Compute f(x)
    f_x = f1 + f2 + f3 + f4

    # Compute partial derivatives
    pdwr_density = (1/delta_l)*(x[:, 1, i_range - 1, j_range, :] - 2*x[:, 1, i_range, j_range, :] + x[:, 1, i_range + 1, j_range, :])
    + (1/delta_l)*(x[:, 2, i_range, j_range - 1, :] - 2*x[:, 2, i_range, j_range, :] + x[:, 2, i_range, j_range + 1, :])
    pdwr_velx = (1/delta_l)*(x[:, 0, i_range - 1, j_range, :] - 2*x[:, 0, i_range, j_range, :] + x[:, 0, i_range + 1, j_range, :])
    pdwr_vely = (1/delta_l)*(x[:, 0, i_range, j_range - 1, :] - 2*x[:, 0, i_range, j_range, :] + x[:, 0, i_range, j_range + 1, :] )

    # Compute the final gradient for the inner grid
    grad_E[:, 0, i_range, j_range, :] = f_x * pdwr_density
    grad_E[:, 1, i_range, j_range, :] = f_x * pdwr_velx
    grad_E[:, 2, i_range, j_range, :] = f_x * pdwr_vely

    return grad_E