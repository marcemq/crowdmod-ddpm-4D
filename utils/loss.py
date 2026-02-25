import torch
def divKLPoissonLoss(rho_hat, rho_gt):
    loss = rho_gt * (torch.log(rho_gt) - torch.log(rho_hat)) + rho_hat - rho_gt
    return loss

def mseLoss(mu_hat, var_hat, mu_gt, var_gt):
    loss = ((mu_hat-mu_gt)*(mu_hat-mu_gt)) +  ((var_hat-var_gt)*(var_hat-var_gt)) # MSE for mu and var
    return loss

def divKLGaussianLoss(mu_hat, var_hat, mu_gt, var_gt):
    div = 1/var_hat
    loss = 0.5*div*((mu_hat-mu_gt)*(mu_hat-mu_gt)) + var_gt*div - torch.log(var_gt*div) - 1
    return loss

def evaluate_loss(model, x, y, teacher_forcing, eps):
    # Forward pass
    yhat = model(x, y, teacher_forcing=teacher_forcing)
    # Estimated density
    rho_hat = torch.exp(yhat[:,0:1,:,:,:]).clamp(min=1e-8, max=20)
    # Ground truth density
    rho_gt  = y[:,0:1,:,:,:].clamp(min=1e-8, max=20)
    # Poisson loss
    rloss   = divKLPoissonLoss(rho_hat, rho_gt)
    rloss   = rloss.mean()

    # Estimated velocity means
    mu_hat  = yhat[:,1:3,:,:,:]
    # Ground truth velocity means
    mu_gt   = y[:,1:3,:,:,:]
    # Estimated velocity variances
    var_hat = torch.exp(yhat[:,3:4,:,:,:]).clamp(min=1e-8, max=20)
    # Estimated velocity variances
    var_gt  = y[:,3:4,:,:,:].clamp(min=1e-8, max=20)
    # velocity loss: MSE or divKLGaussian
    occupied_mask = (rho_gt >= 1.0).float()
    empty_mask = 1.0 - occupied_mask

    occupied_count = occupied_mask.sum()
    empty_count = empty_mask.sum()
    # Expand mask for velocity channels
    occupied_mask = occupied_mask.repeat(1, 2, 1, 1, 1)
    # Supervised velocity loss (occupied regions)
    mse_vloss   = mseLoss(mu_hat, var_hat, mu_gt, var_gt)
    loss_considering_density = (occupied_mask * mse_vloss).sum() / (occupied_count + eps)
    # Empty-region regularization
    vel_norm = mu_hat[:,0:1]**2 + mu_hat[:,1:2]**2
    var_penalty = var_hat*var_hat
    loss_not_considering_density = (empty_mask * (vel_norm + var_penalty)).sum() / (empty_count + eps)
    # Total velocity loss
    vloss = loss_considering_density + loss_not_considering_density

    return rloss, vloss, loss_considering_density, loss_not_considering_density