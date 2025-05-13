import argparse
import sys
import os, re
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import logging
import torch
import sys
import wandb
import numpy as np
from torchvision.utils import make_grid
import torch.optim as optim
import gc,logging,os

from matplotlib import pyplot as plt
from utils.myparser import getYamlConfig
from utils.dataset import getDataset, getClassicDataset
from utils.model_details import count_trainable_params
from models.diffusion.forward import ForwardSampler
from models.unet import MacropropsDenoiser
from models.diffusion.ddpm import DDPM
from models.training import train_one_epoch
from functools import partial

def train(cfg, filenames, show_losses_plot=False):
    wandb.init(
        project="macroprops-predict-4D",
        config={
        "architecture": "DDPM-4D",
        "dataset": cfg.DATASET.NAME,
        "learning_rate": cfg.TRAIN.SOLVER.LR,
        "epochs": cfg.TRAIN.EPOCHS,
        "batch_size": cfg.DATASET.BATCH_SIZE,
        "past_len": cfg.DATASET.PAST_LEN,
        "future_len": cfg.DATASET.FUTURE_LEN,
        "weight_decay": cfg.TRAIN.SOLVER.WEIGHT_DECAY,
        "solver_betas": cfg.TRAIN.SOLVER.BETAS,
        }
    )
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get batched datasets ready to iterate
    if cfg.DATASET.CLASSIC_SPLIT:
        batched_train_data, _ = getClassicDataset(cfg, filenames)
    else:
        batched_train_data, _, _ = getDataset(cfg, filenames, train_data_only=True)

    logging.info(f"Batched Train dataset loaded.")
    # Instanciate the UNet for the reverse diffusion
    denoiser = MacropropsDenoiser(input_channels  = cfg.MACROPROPS.MPROPS_COUNT,
                                  output_channels = cfg.MACROPROPS.MPROPS_COUNT,
                                  num_res_blocks  = cfg.MODEL.NUM_RES_BLOCKS,
                                  base_channels           = cfg.MODEL.BASE_CH,
                                  base_channels_multiples = cfg.MODEL.BASE_CH_MULT,
                                  apply_attention         = cfg.MODEL.APPLY_ATTENTION,
                                  dropout_rate            = cfg.MODEL.DROPOUT_RATE,
                                  time_multiple           = cfg.MODEL.TIME_EMB_MULT,
                                  condition               = cfg.MODEL.CONDITION)
    denoiser.to(device)
    trainable_params = count_trainable_params(denoiser)
    logging.info(f"Total trainable parameters at denoiser:{trainable_params}")
    # The optimizer (Adam with weight decay)
    optimizer = optim.Adam(denoiser.parameters(),lr=cfg.TRAIN.SOLVER.LR, betas=cfg.TRAIN.SOLVER.BETAS,weight_decay=cfg.TRAIN.SOLVER.WEIGHT_DECAY)

    # Instantiate the diffusion model
    diffusionmodel = DDPM(timesteps=cfg.DIFFUSION.TIMESTEPS, scale=cfg.DIFFUSION.SCALE)
    diffusionmodel.to(device)


    best_loss      = 1e6
    consecutive_nan_count = 0
    # Training loop
    for epoch in range(1,cfg.TRAIN.EPOCHS + 1):
        torch.cuda.empty_cache()
        gc.collect()

        # One epoch of training
        epoch_loss = train_one_epoch(denoiser,diffusionmodel,batched_train_data,optimizer,device,epoch=epoch,total_epochs=cfg.TRAIN.EPOCHS)
        wandb.log({"loss": epoch_loss})
        # Check for consecutives nans
        if np.isnan(epoch_loss):
            consecutive_nan_count += 1
            if consecutive_nan_count >=3:
                wandb.finish()
                break

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
            lr_str = "{:.0e}".format(cfg.TRAIN.SOLVER.LR)
            save_path = cfg.MODEL.SAVE_DIR+(cfg.MODEL.MODEL_NAME.format(cfg.TRAIN.EPOCHS, lr_str, cfg.DATASET.TRAIN_FILE_COUNT, cfg.DATASET.PAST_LEN, cfg.DATASET.FUTURE_LEN))
            torch.save(checkpoint_dict, save_path)
            del checkpoint_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train a diffusion model for crowd macroproperties.")
    parser.add_argument('--config-yml-file', type=str, default='config/ATC_ddpm_4test.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/ATC_ddpm_DSlist4test.yml',help='Configuration YML macroprops list for specific dataset.')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    filenames = cfg.DATA_LIST

    if cfg.DATASET.NAME in ["ATC", "ATC4TEST"]:
        filenames = [filename.replace(".csv", ".pkl") for filename in filenames]
    elif cfg.DATASET.NAME in ["HERMES-BO"]:
        filenames = [filename.replace(".txt", ".pkl") for filename in filenames]
    else:
        logging.info("Dataset not supported")

    filenames = [ os.path.join(cfg.PICKLE.PICKLE_DIR, filename) for filename in filenames if filename.endswith('.pkl')]
    train(cfg, filenames)
