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
from utils.utils import create_directory
from utils.myparser import getYamlConfig
from utils.dataset import getDataset, getClassicDataset
from utils.model_details import count_trainable_params
from models.diffusion.forward import ForwardSampler
from models.unet import MacropropsDenoiser
from models.diffusion.ddpm import DDPM
from models.training import train_one_epoch, train_one_epoch_convGRU
from models.convGRU.forecaster import Forecaster
from functools import partial

def save_checkpoint(optimizer, model, epoch, cfg, arch, best_flag=False):
    checkpoint_dict = {
        "opt": optimizer.state_dict(),
        "model": model.state_dict()
    }

    if best_flag:
        save_path = cfg.DATA_FS.SAVE_DIR+(cfg.MODEL.NAME.format(arch, cfg.TRAIN.EPOCHS, cfg.DATASET.PAST_LEN, cfg.DATASET.FUTURE_LEN, "000", cfg.DATASET.VELOCITY_NORM))
    else:
        save_path = cfg.DATA_FS.SAVE_DIR+(cfg.MODEL.NAME.format(arch, cfg.TRAIN.EPOCHS, cfg.DATASET.PAST_LEN, cfg.DATASET.FUTURE_LEN, epoch, cfg.DATASET.VELOCITY_NORM))

    torch.save(checkpoint_dict, save_path)
    del checkpoint_dict

def train_ddpm(cfg, filenames, arch):
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get batched datasets ready to iterate
    if cfg.DATASET.DATASET_TYPE == "BySplitRatio":
        batched_train_data, _ = getClassicDataset(cfg, filenames)
    elif cfg.DATASET.DATASET_TYPE == "ByFilenames":
        batched_train_data, _, _ = getDataset(cfg, filenames, train_data_only=True)
    else:
        logging.error(f"Dataset type not supported.")

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
            save_checkpoint(optimizer, denoiser, epoch, cfg, arch, best_flag=True)

        # Save model samples at stable loss
        if epoch in epoch_model_samples:
            logging.info(f"Epoch {epoch}: in sample set, saving model sample.")
            save_checkpoint(optimizer, denoiser, epoch, cfg, arch)

def train_convGRU(cfg, filenames, arch):
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get batched datasets ready to iterate
    if cfg.DATASET.DATASET_TYPE == "BySplitRatio":
        batched_train_data, batched_val_data = getClassicDataset(cfg, filenames)
    elif cfg.DATASET.DATASET_TYPE == "ByFilenames":
        batched_train_data, batched_val_data, _ = getDataset(cfg, filenames)
    else:
        logging.error(f"Dataset type not supported.")

    logging.info(f"Batched Traininig  and Validation dataset loaded.")

    convGRU_model = Forecaster(input_size  = (cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS),
                               input_channels       = cfg.MACROPROPS.MPROPS_COUNT,
                               enc_hidden_channels  = cfg.MODEL.CONVGRU.ENC_HIDDEN_CH,
                               forc_hidden_channels = cfg.MODEL.CONVGRU.FORC_HIDDEN_CH,
                               enc_kernels          = cfg.MODEL.CONVGRU.ENC_KERNELS,
                               forc_kernels         = cfg.MODEL.CONVGRU.FORC_KERNELS,
                               device               = device,
                               bias                 = False)
    
    # The optimizer (Adam with weight decay)
    optimizer = optim.Adam(convGRU_model.parameters(),lr=cfg.TRAIN.SOLVER.LR, betas=cfg.TRAIN.SOLVER.BETAS,weight_decay=cfg.TRAIN.SOLVER.WEIGHT_DECAY)
    best_loss      = 1e6
    consecutive_nan_count = 0
    # Training loop
    for epoch in range(1,cfg.TRAIN.EPOCHS + 1):
        torch.cuda.empty_cache()
        gc.collect()
        epoch_train_loss, epoch_val_loss = train_one_epoch_convGRU(convGRU_model,batched_train_data, batched_val_data, optimizer, device, epoch=epoch, total_epochs=cfg.TRAIN.EPOCHS, teacher_forcing=cfg.MODEL.CONVGRU.TEACHER_FORCING)
        wandb.log({
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss
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
            save_checkpoint(optimizer, convGRU_model, epoch, cfg, arch, best_flag=True)

def training_mgmt(args, cfg):
    """
    Training management function.
    """
    filenames = cfg.DATA_LIST

    if cfg.DATASET.NAME in ["ATC", "ATC4TEST"]:
        filenames = [filename.replace(".csv", ".pkl") for filename in filenames]
    elif cfg.DATASET.NAME in ["HERMES-BO", "HERMES-CR-120", "HERMES-CR-120-OBST"]:
        filenames = [filename.replace(".txt", ".pkl") for filename in filenames]
    else:
        logging.info("Dataset not supported")

    filenames = [ os.path.join(cfg.DATA_FS.PICKLE_DIR, filename) for filename in filenames if filename.endswith('.pkl')]
    create_directory(cfg.DATA_FS.SAVE_DIR)

    # Initialize W&B
    wandb.init(
        project="macroprops-predict-4D",
        config={
            "architecture": args.arch,
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

    if args.arch == "DDPM-UNet":
        train_ddpm(cfg, filenames, arch=args.arch)
    elif args.arch == "ConvGRU":
        train_convGRU(cfg, filenames, arch=args.arch)
    else:
        logging.info("Architecture not supported.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train a diffusion model for crowd macroproperties.")
    parser.add_argument('--config-yml-file', type=str, default='config/4test/ATC_ddpm.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/4test/ATC_ddpm_datafiles.yml',help='Configuration YML macroprops list for specific dataset.')
    parser.add_argument('--arch', type=str, default='DDPM-UNet', help='Architecture to be used, options: DDPM-UNet|ConvGRU')
    args = parser.parse_args()
    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    training_mgmt(args, cfg)
