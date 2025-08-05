import argparse
import sys
import os, re
import logging
import torch
import pickle
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from utils.utils import create_directory
from utils.myparser import getYamlConfig
from utils.dataset import dataHelper

def save_pickle_file(data, name, sdata_path):
    logging.info(f"New {name} tensor shape:{data.shape}")
    try:
        with open(f"{sdata_path}/{name}.pkl", 'wb') as file:
            pickle.dump(data, file)
    except MemoryError:
        logging.info(f"MemoryError: Unable to pickle {name} due to memory issues.")

def get_synthetic_forward(B, H, W, L, vel_x=0.8):
    """
    Create synthetic tensor by adding a pedestrian from left to right
    """
    pedestrians_forward = np.zeros((1, 3, H, W, L))
    for l in range(min(L, W)):
        pedestrians_forward[0, 0, 6, l, l] = 1
        pedestrians_forward[0, 1, 6, l, l] = vel_x
        pedestrians_forward[0, 2, 6, l, l] = 0.0
    # Expand tensor to match batch size
    return np.tile(pedestrians_forward, (B, 1, 1, 1, 1))

def get_synthetic_backward(B, H, W, L, vel_x=0.8):
    """
    Create synthetic tensor by adding a pedestrian from right to left
    """
    pedestrians_backward = np.zeros((1, 3, H, W, L))
    for l in range(min(L, W)):
        pedestrians_backward[0, 0, 6, W-1-l, l] = 1
        pedestrians_backward[0, 1, 6, W-1-l, l] = -vel_x
        pedestrians_backward[0, 2, 6, W-1-l, l] = 0.0
    # Expand tensor to match batch size
    return np.tile(pedestrians_backward, (B, 1, 1, 1, 1))

def generate_synthetic_data(cfg, filenames, samples_synthetic, type_synthetic):
    sdata_path = os.path.join(os.getcwd(), "datasets", cfg.DATASET.NAME+"_SYNTHETIC")
    create_directory(sdata_path)
    np.random.seed(42)
    # Get macroprps raw tensor
    _, _, tmp_test_data, _, _, _ = dataHelper(cfg, filenames, cfg.MACROPROPS.MPROPS_COUNT, train_data_only=False, test_data_only=True)
    shuffled_indices = np.random.permutation(tmp_test_data.shape[0])[:samples_synthetic]
    true_data = tmp_test_data[shuffled_indices].copy()
    save_pickle_file(true_data, "true_data", sdata_path)

    synthetic_data = true_data.copy()
    B, _, H, W, L = synthetic_data.shape

    synthetic_forward, synthetic_backward = [],[]
    if type_synthetic in ['FORWARD', 'ALL']:
        synthetic_forward = get_synthetic_forward(B, H, W, L)
    if type_synthetic in ['BACKWARD', 'ALL']:
        synthetic_backward = get_synthetic_backward(B, H, W, L)

    # # Element-wise addition
    synthetic_data += (synthetic_forward + synthetic_backward)
    save_pickle_file(synthetic_data, "synthetic_data", sdata_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train a diffusion model for crowd macroproperties.")
    parser.add_argument('--config-yml-file', type=str, default='config/ATC_ddpm.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/ATC_ddpm_DSlist.yml',help='Configuration YML macroprops list for specific dataset.')
    parser.add_argument('--samples-synthetic', type=int, default=20,help='Samples of synthetic sequences to be produced')
    parser.add_argument('--type-synthetic', type=str, default='ALL',help='Type of synthetic data to be added FORWARD|BACKWARD|ALL')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    filenames = cfg.DATA_LIST
    filenames = [filename.replace(".csv", ".pkl") for filename in filenames]
    filenames = [ os.path.join(cfg.DATA_FS.PICKLE_DIR, filename) for filename in filenames if filename.endswith('.pkl')]
    generate_synthetic_data(cfg, filenames, args.samples_synthetic, args.type_synthetic)