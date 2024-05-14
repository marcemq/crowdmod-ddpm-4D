import argparse
import sys
import os, re
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
import torch.nn as nn
from utils.myparser import getYamlConfig
from models.unet import MacropropsDenoiser
from models.diffusion.ddpm import DDPM

def num_params(module:nn.Module):
    n = 0
    for m in module.parameters():
        if m.requires_grad:
            n += m.numel()
    return n

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to show the total of trainable parameters")
    parser.add_argument('--config-yml-file', type=str, default='config/ATC_ddpm.yml', help='Configuration YML file for specific dataset.')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    denoiser_model = MacropropsDenoiser(num_res_blocks = cfg.MODEL.NUM_RES_BLOCKS,
                                base_channels           = cfg.MODEL.BASE_CH,
                                base_channels_multiples = cfg.MODEL.BASE_CH_MULT,
                                apply_attention         = cfg.MODEL.APPLY_ATTENTION,
                                dropout_rate            = cfg.MODEL.DROPOUT_RATE,
                                time_multiple           = cfg.MODEL.TIME_EMB_MULT,
                                condition               = cfg.MODEL.CONDITION)

    diffusion_model = DDPM(timesteps=cfg.DIFFUSION.TIMESTEPS, scale=cfg.DIFFUSION.SCALE)

    denoiser_model.to(device)
    diffusion_model.to(device)

    print(f' Total trainable params in denoiser model: {num_params(denoiser_model)}')
    print(f' Total trainable params in diffusion model: {num_params(diffusion_model)}')