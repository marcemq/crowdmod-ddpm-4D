import torch
def divKLPoissonLoss(rho_hat, rho_gt):
    loss = rho_gt * (torch.log(rho_gt) - torch.log(rho_hat)) + rho_hat - rho_gt
    return loss

def divKLGaussianLoss(mu_hat, var_hat, mu_gt, var_gt):
    div = 1/var_hat
    #loss = 0.5*div*((mu_hat-mu_gt)*(mu_hat-mu_gt)) + var_gt*div - torch.log(var_gt*div) - 1
    loss = ((mu_hat-mu_gt)*(mu_hat-mu_gt)) # MSE for mu
    #loss = ((mu_hat-mu_gt)*(mu_hat-mu_gt)) +  ((var_hat-var_gt)*(var_hat-var_gt)) # MSE for mu and var
    #loss = ((mu_hat-mu_gt)*(mu_hat-mu_gt)) / var_hat + torch.log(var_hat)
    #loss = 0.5*div*((mu_hat-mu_gt)*(mu_hat-mu_gt)) - torch.log(var_gt*div)
    return loss

def evaluate_loss(model, x, y, teacher_forcing):
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
    # Gaussian KL loss
    vloss   = divKLGaussianLoss(mu_hat, var_hat, mu_gt, var_gt)
    #vloss = vloss.mean()
    vloss   = (rho_gt.repeat(1, 2, 1, 1, 1)*vloss).mean()

    return rloss, vloss

def evaluate_loss_bu(model, x, y, teacher_forcing):
    # Forward pass
    yhat = model(x, y, teacher_forcing=teacher_forcing)
    # Estimated density
    rho_hat = torch.exp(yhat[:,0:1,:,:,:]).clamp(min=1e-8, max=20)
    # Ground truth density
    rho_gt  = y[:,0:1,:,:,:].clamp(min=1e-8, max=20)
    # Poisson loss
    rloss = torch.nn.PoissonNLLLoss()
    out_rloss = rloss(rho_hat, rho_gt)
    out_rloss   = out_rloss.mean()

    # Estimated velocity means
    mu_hat  = yhat[:,1:3,:,:,:]
    # Ground truth velocity means
    mu_gt   = y[:,1:3,:,:,:]
    # Estimated velocity variances
    var_hat = torch.exp(yhat[:, 3:4, :, :, :]).clamp(min=1e-8, max=20)
    var_hat = var_hat.expand_as(mu_hat)
    # Estimated velocity variances
    var_gt  = y[:,3:4,:,:,:].clamp(min=1e-8, max=20)
    # Gaussian KL loss
    vloss = torch.nn.GaussianNLLLoss()
    out_vloss = vloss(mu_hat, mu_gt, var_hat)
    out_vloss   = (rho_gt.repeat(1, 2, 1, 1, 1)*out_vloss).mean()

    return out_rloss, out_vloss