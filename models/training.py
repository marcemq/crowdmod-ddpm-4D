import torch
import torch.nn as nn   
import torch.nn.functional as F
from torch.cuda import amp
from torchmetrics import MeanMetric
from tqdm import tqdm
from models.diffusion.ddpm import DDPM


# Apply one training step
def train_step(future:torch.Tensor,past:torch.Tensor, denoiser_model:nn.Module, forwardsampler:DDPM):
    # Sample a timestep uniformly
    t = torch.randint(low=0, high=forwardsampler.timesteps, size=(future.shape[0],), device=future.device)
    # Apply forward noising process on original images, up to step t (sample from q(x_t|x_0))
    future_macroprops_noisy, eps_true = forwardsampler(future, t)
    with amp.autocast():
        # Our prediction for the denoised macropros sequence AR:beLOW is needed a permutation? lo we have LxHxW?
        # TODO: eps_predicted debe de ser de la dimension de eps_true
        eps_predicted = denoiser_model(future_macroprops_noisy, t, past)
        # Deduce the loss
        loss          = F.mse_loss(eps_predicted, eps_true)
    return loss

# One epoch of training
def train_one_epoch(denoiser_model:nn.Module,sampler:nn.Module,loader,optimizer,device,epoch,total_epochs):

    loss_record = MeanMetric()
    # Set in training mode
    denoiser_model.train()

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{total_epochs}")
        # Scan the batches
        for batched_train_data in loader:
            tq.update(1)
            # Take a batch of macropros sequences
            past_train, future_train, stats = batched_train_data
            past_train, future_train = past_train.float(), future_train.float()
            past_train, future_train = past_train.to(device=device), future_train.to(device=device)

            # Evaluate loss AR: I need to pass the GT to below? of the loss be computed taking only x_train?
            loss = train_step(future_train, past_train, denoiser_model, sampler)

            # Backpropagation and update
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

        mean_loss = loss_record.compute().item()

        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

    return mean_loss