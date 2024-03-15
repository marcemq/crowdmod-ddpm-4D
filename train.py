import sys
import os, re
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import logging
import torch
import sys
import wandb
from torchvision.utils import make_grid
import torch.optim as optim
import gc,logging,os

from matplotlib import pyplot as plt
from utils.myparser import getYamlConfig
from utils.dataset import getDataset
from models.diffusion.forward import ForwardSampler
from models.unet import MacroprosDenoiser
from models.diffusion.ddpm import DDPM
from models.training import train_one_epoch
from torchsummary import summary
from functools import partial

def train(cfg, filenames, show_losses_plot=False):
    wandb.init(
        project="macroprops-predict",
        config={
        "architecture": "DDPM",
        "dataset": cfg.DATASET.NAME,
        "learning_rate": cfg.TRAIN.SOLVER.LR,
        "epochs": cfg.TRAIN.EPOCHS,
        "batch_size": cfg.DATASET.params.batch_size,
        "observation_len": cfg.DATASET.OBS_LEN,
        "prediction_len": cfg.DATASET.PRED_LEN,
        "weight_decay": cfg.TRAIN.SOLVER.WEIGHT_DECAY,
        "solver_betas": cfg.TRAIN.SOLVER.BETAS,
        }
    )
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get batched datasets ready to iterate
    batched_train_data, _, _ = getDataset(cfg, filenames)

    # Instanciate the UNet for the reverse diffusion
    denoiser = MacroprosDenoiser(num_res_blocks = cfg.MODEL.NUM_RES_BLOCKS,
                                base_channels           = cfg.MODEL.BASE_CH,
                                base_channels_multiples = cfg.MODEL.BASE_CH_MULT,
                                apply_attention         = cfg.MODEL.APPLY_ATTENTION,
                                dropout_rate            = cfg.MODEL.DROPOUT_RATE,
                                time_multiple           = cfg.MODEL.TIME_EMB_MULT)
    denoiser.to(device)
    #specific_timesteps = [250]
    #t = torch.as_tensor(specific_timesteps, dtype=torch.long)
    #t = torch.randint(low=0, high=1000, size=(64,), device=device)
    #summary(denoiser, [(64, 4, 12, 36, 5), t] )

    # The optimizer (Adam with weight decay)
    optimizer = optim.Adam(denoiser.parameters(),lr=cfg.TRAIN.SOLVER.LR, betas=cfg.TRAIN.SOLVER.BETAS,weight_decay=cfg.TRAIN.SOLVER.WEIGHT_DECAY)

    # Instantiate the diffusion model
    diffusionmodel = DDPM(timesteps=cfg.DIFFUSION.TIMESTEPS, scale=cfg.DIFFUSION.SCALE)
    diffusionmodel.to(device)

    # Training loop
    best_loss      = 1e6
    for epoch in range(1,cfg.TRAIN.EPOCHS + 1):
        torch.cuda.empty_cache()
        gc.collect()

        # One epoch of training
        epoch_loss = train_one_epoch(denoiser,diffusionmodel,batched_train_data,optimizer,device,epoch=epoch,total_epochs=cfg.TRAIN.EPOCHS)
        wandb.log({"loss_2D": epoch_loss})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Save best checkpoints -> AR, shouldn't we save diffusionmodel too?? I think it also has weigths, isn't?
            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "model": denoiser.state_dict()
            }
            if not os.path.exists(cfg.MODEL.SAVE_DIR):
                # Create a new directory if it does not exist
                os.makedirs(cfg.MODEL.SAVE_DIR)
            lr_parts = str(cfg.TRAIN.SOLVER.LR).split('.')
            scale_parts = str(cfg.DIFFUSION.SCALE).split('.')
            save_path = cfg.MODEL.SAVE_DIR+(cfg.MODEL.MODEL_NAME.format(cfg.TRAIN.EPOCHS, lr_parts[0], scale_parts[1]))
            torch.save(checkpoint_dict, save_path)
            del checkpoint_dict

if __name__ == '__main__':
    cfg = getYamlConfig()
    filenames = cfg.SUNDAY_DATA_LIST
    filenames = [filename.replace(".csv", ".pkl") for filename in filenames]
    filenames = [ os.path.join(cfg.PICKLE.PICKLE_DIR, filename) for filename in filenames if filename.endswith('.pkl')]
    train(cfg, filenames)
