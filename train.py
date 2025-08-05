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

def save_checkpoint(optimizer, denoiser, epoch, cfg, best_flag=False):
    checkpoint_dict = {
        "opt": optimizer.state_dict(),
        "model": denoiser.state_dict()
    }

    if best_flag:
        save_path = cfg.DATA_FS.SAVE_DIR+(cfg.MODEL.NAME.format("UNet", cfg.TRAIN.EPOCHS, cfg.DATASET.PAST_LEN, cfg.DATASET.FUTURE_LEN, "best", cfg.DATASET.VELOCITY_NORM))
    else:
        save_path = cfg.DATA_FS.SAVE_DIR+(cfg.MODEL.NAME.format("UNet", cfg.TRAIN.EPOCHS, cfg.DATASET.PAST_LEN, cfg.DATASET.FUTURE_LEN, epoch, cfg.DATASET.VELOCITY_NORM))

    torch.save(checkpoint_dict, save_path)
    del checkpoint_dict

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
    # Create a new directory if it does not exist
    if not os.path.exists(cfg.DATA_FS.SAVE_DIR):
        os.makedirs(cfg.DATA_FS.SAVE_DIR)

    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get batched datasets ready to iterate
    if cfg.DATASET.DATASET_TYPE == "BySplitRatio":
        batched_train_data, _ = getClassicDataset(cfg, filenames)
    elif cfg.DATASET.DATASET_TYPE == "ByFilenames":
        batched_train_data, _, _ = getDataset(cfg, filenames, train_data_only=True)

    logging.info(f"Batched Train dataset loaded.")
    # Instanciate the UNet for the reverse diffusion
    denoiser = MacropropsDenoiser(input_channels  = cfg.MACROPROPS.MPROPS_COUNT,
                                  output_channels = cfg.MACROPROPS.MPROPS_COUNT,
                                  num_res_blocks  = cfg.MODEL.DDPM.UNET.NUM_RES_BLOCKS,
                                  base_channels           = cfg.MODEL.DDPM.UNET.BASE_CH,
                                  base_channels_multiples = cfg.MODEL.DDPM.UNET.BASE_CH_MULT,
                                  apply_attention         = cfg.MODEL.DDPM.UNET.APPLY_ATTENTION,
                                  dropout_rate            = cfg.MODEL.DDPM.UNET.DROPOUT_RATE,
                                  time_multiple           = cfg.MODEL.DDPM.UNET.TIME_EMB_MULT,
                                  condition               = cfg.MODEL.DDPM.UNET.CONDITION)
    denoiser.to(device)
    trainable_params = count_trainable_params(denoiser)
    logging.info(f"Total trainable parameters at denoiser:{trainable_params}")
    # The optimizer (Adam with weight decay)
    optimizer = optim.Adam(denoiser.parameters(),lr=cfg.TRAIN.SOLVER.LR, betas=cfg.TRAIN.SOLVER.BETAS,weight_decay=cfg.TRAIN.SOLVER.WEIGHT_DECAY)

    # Instantiate the diffusion model
    diffusionmodel = DDPM(timesteps=cfg.MODEL.DDPM.TIMESTEPS, scale=cfg.MODEL.DDPM.SCALE)
    diffusionmodel.to(device)


    best_loss      = 1e6
    consecutive_nan_count = 0
    epoch_model_samples = np.random.randint(150, cfg.TRAIN.EPOCHS + 1, size=cfg.MODEL.DDPM.MODEL_SAMPLES)
    # Training loop
    for epoch in range(1,cfg.TRAIN.EPOCHS + 1):
        torch.cuda.empty_cache()
        gc.collect()

        # One epoch of training
        epoch_loss = train_one_epoch(denoiser,diffusionmodel,batched_train_data,optimizer,device,epoch=epoch,total_epochs=cfg.TRAIN.EPOCHS)
        wandb.log({"loss": epoch_loss})
        # NaN handling / early stopping
        if np.isnan(epoch_loss):
            consecutive_nan_count += 1
            logging.warning(f"Epoch {epoch}: loss is NaN ({consecutive_nan_count} consecutive)")
            if consecutive_nan_count >= 3:
                logging.error("Loss has been NaN for 3 consecutive epochs; terminating training early.")
                wandb.finish()
                break
        else:
            consecutive_nan_count = 0  # reset on valid loss

        # Save best checkpoint from all training
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(optimizer, denoiser, epoch, cfg, best_flag=True)

        # Save model samples at stable loss
        if epoch in epoch_model_samples:
            logging.info(f"Epoch {epoch}: in sample set, saving model sample.")
            save_checkpoint(optimizer, denoiser, epoch, cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train a diffusion model for crowd macroproperties.")
    parser.add_argument('--config-yml-file', type=str, default='config/ATC_ddpm_4test.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/ATC_ddpm_DSlist4test.yml',help='Configuration YML macroprops list for specific dataset.')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    filenames = cfg.DATA_LIST

    if cfg.DATASET.NAME in ["ATC", "ATC4TEST"]:
        filenames = [filename.replace(".csv", ".pkl") for filename in filenames]
    elif cfg.DATASET.NAME in ["HERMES-BO", "HERMES-CR-120", "HERMES-CR-120-OBST"]:
        filenames = [filename.replace(".txt", ".pkl") for filename in filenames]
    else:
        logging.info("Dataset not supported")

    filenames = [ os.path.join(cfg.DATA_FS.PICKLE_DIR, filename) for filename in filenames if filename.endswith('.pkl')]
    train(cfg, filenames)
