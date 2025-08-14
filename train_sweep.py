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
from utils.model_details import count_trainable_params
from utils.utils import get_filenames_paths, get_training_dataset, get_sweep_configuration, save_checkpoint, init_wandb, create_directory
from models.diffusion.forward import ForwardSampler
from models.unet import MacropropsDenoiser
from models.diffusion.ddpm import DDPM
from models.training import train_one_epoch, train_one_epoch_convGRU
from models.convGRU.forecaster import Forecaster
from torchsummary import summary
from functools import partial

def train_sweep_ddpm(cfg, filenames, show_losses_plot=False):
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
    denoiser = MacropropsDenoiser(input_channels = cfg.MACROPROPS.MPROPS_COUNT,
                                  output_channels = cfg.MACROPROPS.MPROPS_COUNT,
                                  num_res_blocks = cfg.MODEL.DDPM.UNET.NUM_RES_BLOCKS,
                                  base_channels           = wandb.config.base_ch,
                                  base_channels_multiples = cfg.MODEL.DDPM.UNET.BASE_CH_MULT,
                                  apply_attention         = cfg.MODEL.DDPM.UNET.APPLY_ATTENTION,
                                  dropout_rate            = wandb.config.dropout_rate,
                                  time_multiple           = wandb.config.time_emb_mult,
                                  condition               = cfg.MODEL.DDPM.UNET.CONDITION)
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
            if not os.path.exists(cfg.DATA_FS.SAVE_DIR):
                # Create a new directory if it does not exist
                os.makedirs(cfg.DATA_FS.SAVE_DIR)
            lr_str = "{:.0e}".format(wandb.config.learning_rate)
            save_path = cfg.DATA_FS.SAVE_DIR+(cfg.MODEL.NAME.format(wandb.config.epochs, cfg.DATASET.PAST_LEN, cfg.DATASET.FUTURE_LEN))
            torch.save(checkpoint_dict, save_path)
            del checkpoint_dict

def train_sweep_convGRU(cfg, batched_train_data, batched_val_data, arch, mprops_count, project_name):
    init_wandb(cfg, args.arch, project_name)
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Batched Traininig  and Validation dataset loaded.")

    # Set forc_hidden_channels based on sweep enc_hidden_ch
    forc_hidden_map = {(16, 64, 64, 96, 96, 96):  [96, 96, 96, 96, 96, 64, 16],
                       (32, 64, 64, 96, 96, 96):  [96, 96, 96, 96, 96, 64, 32],
                       (16, 64, 64, 128, 128, 128): [128, 128, 128, 128, 128, 64, 16],
                       (32, 64, 64, 128, 128, 128): [128, 128, 128, 128, 128, 64, 32]}
    forc_hidden_ch = forc_hidden_map[tuple(wandb.config.enc_hidden_ch)]

    # Set convGRU model object
    convGRU_model = Forecaster(input_size  = (cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS),
                               input_channels       = mprops_count,
                               enc_hidden_channels  = wandb.config.enc_hidden_ch,
                               forc_hidden_channels = forc_hidden_ch,
                               enc_kernels          = cfg.MODEL.CONVGRU.ENC_KERNELS,
                               forc_kernels         = cfg.MODEL.CONVGRU.FORC_KERNELS,
                               device               = device,
                               bias                 = False)

    trainable_params = count_trainable_params(convGRU_model)
    logging.info(f"Total trainable parameters at ConvGRU model:{trainable_params}")
    # Set optimizer based on chossen wandb sweep optimizer
    if wandb.config.optimizer == "Adam":
        optimizer = optim.Adam(
            convGRU_model.parameters(),
            lr=wandb.config.learning_rate,
            betas=wandb.config.betas,
            weight_decay=wandb.config.weight_decay
        )
    elif wandb.config.optimizer == "AdamW":
        optimizer = optim.AdamW(
            convGRU_model.parameters(),
            lr=wandb.config.learning_rate,
            betas=wandb.config.betas,
            weight_decay=wandb.config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {wandb.config.optimizer}")

    logging.info(f"Selected optimizer: {wandb.config.optimizer}")
    logging.info(f"Selected enc_hidden_ch: {wandb.config.enc_hidden_ch}")
    logging.info(f"Selected forc_hidden_ch: {forc_hidden_ch}")

    best_loss      = 1e6
    # Training loop
    for epoch in range(1, wandb.config.epochs + 1):
        torch.cuda.empty_cache()
        gc.collect()
        epoch_train_loss, epoch_val_loss = train_one_epoch_convGRU(convGRU_model,batched_train_data, batched_val_data, optimizer, device, epoch=epoch, total_epochs=wandb.config.epochs, teacher_forcing=cfg.MODEL.CONVGRU.TEACHER_FORCING)
        wandb.log({
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss
        }, step=epoch)

        if epoch_train_loss < best_loss:
            best_loss = epoch_train_loss
            save_checkpoint(optimizer, convGRU_model, "000", cfg, arch)

def train_sweep_mgmt(args, cfg):
    # === Prepare file paths ===
    filenames = get_filenames_paths(cfg)
    create_directory(cfg.DATA_FS.SAVE_DIR)

    # === Load training dataset
    mprops_count = 4 if args.arch == "ConvGRU" else 3
    batched_train_data, batched_val_data = get_training_dataset(cfg, filenames, mprops_count)

    # === Sweep Train models with specific architecture ===
    sweep_configuration = get_sweep_configuration(args.arch)
    if args.arch == "DDPM-UNet":
        project_name = "sweep_crowdmod_ddpm"
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
        wandb.agent(sweep_id, function=functools.partial(train_sweep_ddpm, cfg, filenames), count=50)
    elif args.arch == "ConvGRU":
        project_name = "sweep_crowdmod_ConvGRU"
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
        wandb.agent(sweep_id, function=functools.partial(train_sweep_convGRU, cfg, batched_train_data, batched_val_data, args.arch, mprops_count, project_name), count=50)
    else:
        logging.error("Architecture not supported to launch train sweep.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train a diffusion model for crowd macroproperties.")
    parser.add_argument('--config-yml-file', type=str, default='config/4test/ATC_ddpm.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/4test/ATC_ddpm_datafiles.yml',help='Configuration YML macroprops list for specific dataset.')
    parser.add_argument('--arch', type=str, default='DDPM-UNet', help='Architecture to be used, options: DDPM-UNet|ConvGRU')
    args = parser.parse_args()
    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    train_sweep_mgmt(args, cfg)
