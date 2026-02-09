import torch
def divKLPoissonLoss(rho_hat, rho_gt):
    loss = rho_gt*torch.log(rho_gt/rho_hat) + rho_hat - rho_gt
    return loss

def divKLGaussianLoss(mu_hat, sigma_hat, mu_gt, sigma_gt):
    #print(f'mu_hat shape:{mu_hat.shape}')
    #print(f'sigma_hat shape:{sigma_hat.shape}')
    #print(f'mu_gt shape:{mu_gt.shape}')
    #print(f'sigma_gt shape:{sigma_gt.shape}')
    div = 1/sigma_hat
    loss = 0.5*div*((mu_hat-mu_gt)*(mu_hat-mu_gt)) + sigma_gt*div - torch.log(sigma_gt*div) - 1
    return loss

def evaluate_loss_bu(model, x, y, teacher_forcing):
    # Forward pass
    yhat = model(x, y, teacher_forcing=teacher_forcing)
    # Estimated density
    rho_hat = torch.clamp(torch.exp(yhat[:,0:1,:,:,:]), min=1e-8, max=20)
    # Ground truth density
    rho_gt  = torch.clamp(y[:,0:1,:,:,:], min=1e-8, max=20)
    # Poisson loss
    rloss   = divKLPoissonLoss(rho_hat, rho_gt)
    rloss   = rloss.mean()
    #logging.debug(f'rloss shape:{rloss.shape}')

    # Estimated velocity means
    mu_hat  = yhat[:,1:3,:,:,:]
    # Ground truth velocity means
    mu_gt   = y[:,1:3,:,:,:]
    # Estimated velocity variances
    var_hat = torch.clamp(torch.exp(yhat[:,3:4,:,:,:]), min=1e-8, max=20)
    # Estimated velocity variances
    var_gt  = torch.clamp(y[:,3:4,:,:,:], min=1e-8, max=20)
    # Gaussian KL loss
    vloss   = divKLGaussianLoss(mu_hat, var_hat, mu_gt, var_gt)
    vloss   = (rho_gt.repeat(1, 2, 1, 1, 1)*vloss).mean()

    # min max rho, var of yhat to have an idea for cliping
    #logging.debug(f"min-max rho:{torch.min(rho_hat):.4f} - {torch.max(rho_hat):.4f}")
    #logging.debug(f"min-max var:{torch.min(var_hat):.4f} - {torch.max(var_hat):.4f}")
    #logging.debug(f"min-max mu_x:{torch.min(yhat[:,1,:,:,:]):.4f} - {torch.max(yhat[:,1,:,:,:]):.4f}")
    #logging.debug(f"min-max mu_y:{torch.min(yhat[:,2,:,:,:]):.4f} - {torch.max(yhat[:,2,:,:,:]):.4f}")
   
    return rloss, vloss

def evaluate_loss(model, x, y, teacher_forcing):
    # Forward pass
    yhat = model(x, y, teacher_forcing=teacher_forcing)
    # Estimated density
    rho_hat = torch.clamp(torch.exp(yhat[:,0:1,:,:,:]), min=1e-8, max=20)
    # Ground truth density
    rho_gt  = torch.clamp(y[:,0:1,:,:,:], min=1e-8, max=20)
    # Poisson loss
    rloss = torch.nn.PoissonNLLLoss()
    out_rloss = rloss(rho_hat, rho_gt, log_input=True)

    # Estimated velocity means
    mu_hat  = yhat[:,1:3,:,:,:]
    # Ground truth velocity means
    mu_gt   = y[:,1:3,:,:,:]
    # Estimated velocity variances
    var_hat = torch.clamp(torch.exp(yhat[:,3:4,:,:,:]), min=1e-8, max=20)
    # Estimated velocity variances
    var_gt  = torch.clamp(y[:,3:4,:,:,:], min=1e-8, max=20)
    # Gaussian KL loss
    vloss = torch.nn.GaussianNLLLoss()
    out_vloss = vloss(mu_hat, mu_gt, var_hat)
    out_vloss   = (rho_gt.repeat(1, 2, 1, 1, 1)*out_vloss).mean()

    return out_rloss, out_vloss