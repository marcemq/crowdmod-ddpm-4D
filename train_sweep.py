import argparse
import sys
import os, re
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import functools

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
from models.unet import MacropropsDenoiser
from models.diffusion.ddpm import DDPM
from models.training import train_one_epoch
from torchsummary import summary
from functools import partial

def train(cfg, filenames, show_losses_plot=False):
    config={
        "architecture": "DDPM-4D",
        "dataset": cfg.DATASET.NAME,
        "learning_rate": cfg.TRAIN.SOLVER.LR,
        "epochs": cfg.TRAIN.EPOCHS,
        "past_len": cfg.DATASET.PAST_LEN,
        "future_len": cfg.DATASET.FUTURE_LEN,
        "weight_decay": cfg.TRAIN.SOLVER.WEIGHT_DECAY,
        "solver_betas": cfg.TRAIN.SOLVER.BETAS,
        }
    wandb.init(project="sweep_crowdmod_ddpm4D")
    # add more params config to wandb
    #wandb.config.update(config)

    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get batched datasets ready to iterate
    batched_train_data, _, _ = getDataset(cfg, filenames, wandb.config.batch_size, train_data_only=True)

    # Instanciate the UNet for the reverse diffusion
    denoiser = MacropropsDenoiser(num_res_blocks = cfg.MODEL.NUM_RES_BLOCKS,
                                base_channels           = wandb.config.base_ch,
                                base_channels_multiples = cfg.MODEL.BASE_CH_MULT,
                                apply_attention         = cfg.MODEL.APPLY_ATTENTION,
                                dropout_rate            = wandb.config.dropout_rate,
                                time_multiple           = wandb.config.time_emb_mult,
                                condition               = cfg.MODEL.CONDITION)
    denoiser.to(device)
    #specific_timesteps = [250]
    #t = torch.as_tensor(specific_timesteps, dtype=torch.long)
    #t = torch.randint(low=0, high=1000, size=(64,), device=device)
    #summary(denoiser, [(64, 4, 12, 36, 5), t] )

    # The optimizer (Adam with weight decay)
    optimizer = optim.Adam(denoiser.parameters(),lr=wandb.config.learning_rate, betas=cfg.TRAIN.SOLVER.BETAS,weight_decay=cfg.TRAIN.SOLVER.WEIGHT_DECAY)

    # Instantiate the diffusion model
    diffusionmodel = DDPM(timesteps=wandb.config.timesteps, scale=wandb.config.scale)
    diffusionmodel.to(device)

    # Training loop
    best_loss      = 1e6
    for epoch in range(1,wandb.config.epochs + 1):
        torch.cuda.empty_cache()
        gc.collect()

        # One epoch of training
        epoch_loss = train_one_epoch(denoiser,diffusionmodel,batched_train_data,optimizer,device,epoch=epoch,total_epochs=wandb.config.epochs)
        wandb.log({"loss": epoch_loss})
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
            lr_str = "{:.0e}".format(wandb.config.learning_rate)
            scale_str = "{:.0e}".format(wandb.config.scale)
            save_path = cfg.MODEL.SAVE_DIR+(cfg.MODEL.MODEL_NAME.format(wandb.config.epochs, lr_str, scale_str, cfg.DATASET.PAST_LEN, cfg.DATASET.FUTURE_LEN))
            torch.save(checkpoint_dict, save_path)
            del checkpoint_dict

sweep_configuration = {
    "name": "sweep_crowdmod_ddpm_4D",
    "method": "random",
    "metric": {"goal": "minimize", "name": "loss_2D"},
    "parameters": {
        "learning_rate": {"min": 0.00001, "max": 0.001},
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [400, 600, 800]},
        "base_ch": {"values": [16, 32, 64]},
        "dropout_rate": {"values": [0.05, 0.15, 0.25]},
        "time_emb_mult": {"values": [2, 4, 8]},
        "scale": {"values": [0.1, 0.3, 0.5, 0.8]},
        "timesteps": {"values": [500, 1000, 1500]},
    },
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train a diffusion model for crowd macroproperties.")
    parser.add_argument('--config-yml-file', type=str, default='config/ATC_ddpm_4test.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/ATC_ddpm_DSlist4test.yml',help='Configuration YML macroprops list for specific dataset.')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    filenames = cfg.SUNDAY_DATA_LIST
    filenames = [filename.replace(".csv", ".pkl") for filename in filenames]
    filenames = [ os.path.join(cfg.PICKLE.PICKLE_DIR, filename) for filename in filenames if filename.endswith('.pkl')]
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="sweep_crowdmod_ddpm4D")
    wandb.agent(sweep_id, function=functools.partial(train, cfg, filenames), count=50)
