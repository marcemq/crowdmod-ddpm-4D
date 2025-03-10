import torch

def sparsityGradient(xnoisy: torch.Tensor, cfg, device) -> torch.Tensor:
    grad = torch.zeros(xnoisy.shape, device=device)
    grad[:,0,:,:]=torch.sign(xnoisy[:,0,:,:])
    # Compute the gradient of the sparsity loss
    return grad

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

import torch

def preservationMassGradient(x, device, delta_t=0.5, delta_l=1.0) -> torch.Tensor:
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

    # Define the inner grid using slices instead of tensors
    i_range = slice(1, H - 1)
    j_range = slice(1, W - 1)

    # First term: temporal finite difference (time axis)
    f1 = (1 / delta_t) * (x[:, 0, i_range, j_range, 1:] - x[:, 0, i_range, j_range, :-1])

    # Second term: spatial interaction term
    f2 = (1 / delta_l) * x[:, 0, i_range, j_range, :] * (
        x[:, 1, i_range.stop - 1, j_range, :] - x[:, 1, i_range.start, j_range, :] +
        x[:, 2, i_range, j_range.stop - 1, :] - x[:, 2, i_range, j_range.start, :]
    )

    # Third term: x[1] interaction
    f3 = (1 / delta_l) * ((x[:, 0, i_range.stop - 1, j_range, :] - x[:, 0, i_range.start, j_range, :]) * 
                           x[:, 1, i_range, j_range, :])

    # Fourth term: x[2] interaction
    f4 = (1 / delta_l) * ((x[:, 0, i_range, j_range.stop - 1, :] - x[:, 0, i_range, j_range.start, :]) * 
                           x[:, 2, i_range, j_range, :])

    # Compute f(x)
    f_x = f1 + f2 + f3 + f4

    # Compute partial derivatives
    pdwr_density = (1 / delta_l) * (
        x[:, 1, i_range.start - 1, j_range, :] - 2 * x[:, 1, i_range, j_range, :] + x[:, 1, i_range.stop, j_range, :]
    ) + (1 / delta_l) * (
        x[:, 2, i_range, j_range.start - 1, :] - 2 * x[:, 2, i_range, j_range, :] + x[:, 2, i_range, j_range.stop, :]
    )

    pdwr_velx = (1 / delta_l) * (
        x[:, 0, i_range.start - 1, j_range, :] - 2 * x[:, 0, i_range, j_range, :] + x[:, 0, i_range.stop, j_range, :]
    )

    pdwr_vely = (1 / delta_l) * (
        x[:, 0, i_range, j_range.start - 1, :] - 2 * x[:, 0, i_range, j_range, :] + x[:, 0, i_range, j_range.stop, :]
    )

    # Compute the final gradient for the inner grid
    grad_E[:, 0, i_range, j_range, :] = f_x * pdwr_density
    grad_E[:, 1, i_range, j_range, :] = f_x * pdwr_velx
    grad_E[:, 2, i_range, j_range, :] = f_x * pdwr_vely

    return grad_E