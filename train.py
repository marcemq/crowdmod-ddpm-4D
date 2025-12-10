import argparse
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import logging
import torch
import sys
from torchvision.utils import make_grid
import logging,os

from utils.utils import create_directory, get_filenames_paths, get_training_dataset, init_wandb
from utils.myparser import getYamlConfig
from utils.model_details import count_trainable_params
from models.diffusion.ddpm import DDPM_model
from models.convGRU.convGRU import ConvGRU_model
from models.flow_matching.flow_matching import FM_model

def train_ddpm(cfg, batched_train_data, arch, mprops_count):
    torch.manual_seed(42)
    ddpm_model = DDPM_model(cfg, arch, mprops_count)
    trainable_params = count_trainable_params(ddpm_model.denoiser)
    logging.info(f"Total trainable parameters at denoiser:{trainable_params}")

    ddpm_model.train(batched_train_data)

def train_fm(cfg, batched_train_data, arch, mprops_count):
    torch.manual_seed(42)
    fm_model = FM_model(cfg, arch, mprops_count)
    trainable_params = count_trainable_params(fm_model.u_predictor)
    logging.info(f"Total trainable parameters at u_predictor:{trainable_params}")

    fm_model.train(batched_train_data)
    
def train_convGRU(cfg, batched_train_data, batched_val_data, arch, mprops_count):
    torch.manual_seed(42)
    convGRU_model = ConvGRU_model(cfg, arch, mprops_count)
    trainable_params = count_trainable_params(convGRU_model.convGRU)
    logging.info(f"Total trainable parameters at ConvGRU model:{trainable_params}")

    convGRU_model.train(batched_train_data, batched_val_data)

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
