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
from utils.utils import create_directory, get_filenames_paths, get_training_dataset, save_checkpoint, init_wandb
from utils.myparser import getYamlConfig
from utils.model_details import count_trainable_params
from models.diffusion.forward import ForwardSampler
from models.unet import MacropropsDenoiser
from models.diffusion.ddpm import DDPM
from models.training import train_one_epoch, train_one_epoch_convGRU, train_one_epoch_fm
from models.convGRU.forecaster import Forecaster
from functools import partial

def train_ddpm(cfg, batched_train_data, arch, mprops_count):
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instanciate the UNet for the reverse diffusion
    denoiser = MacropropsDenoiser(input_channels  = mprops_count,
                                  output_channels = mprops_count,
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
    optimizer = optim.Adam(denoiser.parameters(),lr=cfg.MODEL.DDPM.TRAIN.SOLVER.LR, betas=cfg.MODEL.DDPM.TRAIN.SOLVER.BETAS,weight_decay=cfg.MODEL.DDPM.TRAIN.SOLVER.WEIGHT_DECAY)

    # Instantiate the diffusion model
    diffusionmodel = DDPM(timesteps=cfg.MODEL.DDPM.TIMESTEPS, scale=cfg.MODEL.DDPM.SCALE)
    diffusionmodel.to(device)

    best_loss      = 1e6
    consecutive_nan_count = 0
    low = int(cfg.MODEL.DDPM.TRAIN.EPOCHS * 0.75)
    high = cfg.MODEL.DDPM.TRAIN.EPOCHS + 1  # randint upper bound is exclusive
    epochs_cktp_to_save = np.random.randint(low, high, size=cfg.MODEL.DDPM.CHECKPOINTS_TO_KEEP)
    # Training loop
    for epoch in range(1,cfg.MODEL.DDPM.TRAIN.EPOCHS + 1):
        torch.cuda.empty_cache()
        gc.collect()

        # One epoch of training
        epoch_loss = train_one_epoch(denoiser,diffusionmodel,batched_train_data,optimizer,device,epoch=epoch,total_epochs=cfg.MODEL.DDPM.TRAIN.EPOCHS)
        wandb.log({"train_loss": epoch_loss})
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
            save_checkpoint(optimizer, denoiser, "000", cfg, arch)

        # Save model samples at stable loss
        if epoch in epochs_cktp_to_save:
            logging.info(f"Epoch {epoch}: in sample set, saving model sample.")
            save_checkpoint(optimizer, denoiser, epoch, cfg, arch)

def train_fm(cfg, batched_train_data, arch, mprops_count):
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instanciate the UNet for the reverse diffusion
    unet_model = MacropropsDenoiser(input_channels  = mprops_count,
                                  output_channels = mprops_count,
                                  num_res_blocks  = cfg.MODEL.FLOW_MATCHING.UNET.NUM_RES_BLOCKS,
                                  base_channels           = cfg.MODEL.FLOW_MATCHING.UNET.BASE_CH,
                                  base_channels_multiples = cfg.MODEL.FLOW_MATCHING.UNET.BASE_CH_MULT,
                                  apply_attention         = cfg.MODEL.FLOW_MATCHING.UNET.APPLY_ATTENTION,
                                  dropout_rate            = cfg.MODEL.FLOW_MATCHING.UNET.DROPOUT_RATE,
                                  time_multiple           = cfg.MODEL.FLOW_MATCHING.UNET.TIME_EMB_MULT,
                                  condition               = cfg.MODEL.FLOW_MATCHING.UNET.CONDITION)
    unet_model.to(device)
    trainable_params = count_trainable_params(unet_model)
    logging.info(f"Total trainable parameters at model_unet:{trainable_params}")

    # The optimizer (Adam with weight decay)
    optimizer = optim.Adam(unet_model.parameters(),lr=cfg.MODEL.FLOW_MATCHING.TRAIN.SOLVER.LR, betas=cfg.MODEL.FLOW_MATCHING.TRAIN.SOLVER.BETAS,weight_decay=cfg.MODEL.FLOW_MATCHING.TRAIN.SOLVER.WEIGHT_DECAY)

    best_loss      = 1e6
    consecutive_nan_count = 0
    low = int(cfg.MODEL.FLOW_MATCHING.TRAIN.EPOCHS * 0.75)
    high = cfg.MODEL.FLOW_MATCHING.TRAIN.EPOCHS + 1  # randint upper bound is exclusive
    epochs_cktp_to_save = np.random.randint(low, high, size=cfg.MODEL.FLOW_MATCHING.CHECKPOINTS_TO_KEEP)
    # Training loop
    for epoch in range(1,cfg.MODEL.FLOW_MATCHING.TRAIN.EPOCHS + 1):
        torch.cuda.empty_cache()
        gc.collect()

        # One epoch of training
        epoch_loss = train_one_epoch_fm(unet_model,batched_train_data,optimizer,device,epoch=epoch,total_epochs=cfg.MODEL.FLOW_MATCHING.TRAIN.EPOCHS, time_max_pos=cfg.MODEL.FLOW_MATCHING.TIME_MAX_POS)
        wandb.log({"train_loss": epoch_loss})
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
            save_checkpoint(optimizer, unet_model, "000", cfg, arch)

        # Save model samples at stable loss
        if epoch in epochs_cktp_to_save:
            logging.info(f"Epoch {epoch}: in sample set, saving model sample.")
            save_checkpoint(optimizer, train_one_epoch_fm, epoch, cfg, arch)
    
def train_convGRU(cfg, batched_train_data, batched_val_data, arch, mprops_count):
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Batched Traininig  and Validation dataset loaded.")

    convGRU_model = Forecaster(input_size  = (cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS),
                               input_channels       = mprops_count,
                               enc_hidden_channels  = cfg.MODEL.CONVGRU.ENC_HIDDEN_CH,
                               forc_hidden_channels = cfg.MODEL.CONVGRU.FORC_HIDDEN_CH,
                               enc_kernels          = cfg.MODEL.CONVGRU.ENC_KERNELS,
                               forc_kernels         = cfg.MODEL.CONVGRU.FORC_KERNELS,
                               device               = device,
                               bias                 = False)

    trainable_params = count_trainable_params(convGRU_model)
    logging.info(f"Total trainable parameters at ConvGRU model:{trainable_params}")
    # The optimizer (Adam with weight decay)
    optimizer = optim.Adam(convGRU_model.parameters(),lr=cfg.MODEL.CONVGRU.TRAIN.SOLVER.LR, betas=cfg.MODEL.CONVGRU.TRAIN.SOLVER.BETAS,weight_decay=cfg.MODEL.CONVGRU.TRAIN.SOLVER.WEIGHT_DECAY)
    best_loss      = 1e6
    consecutive_nan_count = 0
    # Training loop
    for epoch in range(1,cfg.MODEL.CONVGRU.TRAIN.EPOCHS + 1):
        torch.cuda.empty_cache()
        gc.collect()
        epoch_train_loss, epoch_val_loss = train_one_epoch_convGRU(convGRU_model,batched_train_data, batched_val_data, optimizer, device, epoch=epoch, total_epochs=cfg.MODEL.CONVGRU.TRAIN.EPOCHS, teacher_forcing=cfg.MODEL.CONVGRU.TEACHER_FORCING)
        wandb.log({
            "train_loss": min(epoch_train_loss, 10),
            "val_loss": min(epoch_val_loss, 10)
        }, step=epoch)
        # NaN handling / early stopping
        if np.isnan(epoch_train_loss):
            consecutive_nan_count += 1
            logging.warning(f"Epoch {epoch}: loss is NaN ({consecutive_nan_count} consecutive)")
            if consecutive_nan_count >= 3:
                logging.error("Loss has been NaN for 3 consecutive epochs; terminating training early.")
                wandb.finish()
                break
        else:
            consecutive_nan_count = 0  # reset on valid loss
        # Save best checkpoint from all training
        if epoch_train_loss < best_loss:
            best_loss = epoch_train_loss
            save_checkpoint(optimizer, convGRU_model, "000", cfg, arch)

def training_mgmt(args, cfg):
    """
    Training management function.
    """
    # === Initialize W&B ===
    init_wandb(cfg, args.arch)

    # === Prepare file paths ===
    filenames_and_numSamples = get_filenames_paths(cfg)
    create_directory(cfg.DATA_FS.SAVE_DIR)

    # === Load training dataset
    mprops_count = 4 if args.arch == "ConvGRU" else 3
    batched_train_data, batched_val_data = get_training_dataset(cfg, filenames_and_numSamples, mprops_count)

    # === Train models with specific architecture ===
    logging.info(f"=======>>>> Init training for {cfg.DATASET.NAME} dataset with {args.arch} architecture.")
    if args.arch == "DDPM-UNet":
        train_ddpm(cfg, batched_train_data, arch=args.arch, mprops_count=mprops_count)
    elif args.arch == "FM-UNet":
        train_fm(cfg, batched_train_data, arch=args.arch, mprops_count=mprops_count)
    elif args.arch == "ConvGRU":
        train_convGRU(cfg, batched_train_data, batched_val_data, arch=args.arch, mprops_count=mprops_count)
    else:
        logging.info("Architecture not supported.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train a diffusion model for crowd macroproperties.")
    parser.add_argument('--config-yml-file', type=str, default='config/4test/ATC_ddpm.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/4test/ATC_ddpm_datafiles.yml',help='Configuration YML macroprops list for specific dataset.')
    parser.add_argument('--arch', type=str, default='DDPM-UNet', help='Architecture to be used, options: DDPM-UNet|FM-UNet|ConvGRU')
    args = parser.parse_args()
    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    training_mgmt(args, cfg)
