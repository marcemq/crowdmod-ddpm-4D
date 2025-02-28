import argparse
import sys
import os, re
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import torch
import pickle
from utils.utils import create_directory
from utils.myparser import getYamlConfig
from utils.dataset import getDataset, dataHelper

def generate_synthetic_data(cfg, filenames,samples_synthetic):
    sdata_path = os.path.join(os.getcwd(), "datasets", cfg.DATASET.NAME+"_SYNTHETIC")
    create_directory(sdata_path)
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get macroprps raw tensor
    _, _, tmp_test_data, _, _, _ = dataHelper(cfg, filenames, cfg.MACROPROPS.MPROPS_COUNT, train_data_only=False, test_data_only=True)
    shuffled_indices = torch.randperm(tmp_test_data.shape[0])
    shuffled_data = tmp_test_data[shuffled_indices]
    synthetic_data = shuffled_data[0:samples_synthetic]
    print(f"synthetic_data tensor shape:{synthetic_data.shape}")
    try:
        with open(sdata_path + "/synthetic_data.pkl", 'wb') as file:
            pickle.dump(synthetic_data, file)
    except MemoryError:
        print("MemoryError: Unable to pickle synthetic data due to memory issues.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to train a diffusion model for crowd macroproperties.")
    parser.add_argument('--config-yml-file', type=str, default='config/ATC_ddpm.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/ATC_ddpm_DSlist.yml',help='Configuration YML macroprops list for specific dataset.')
    parser.add_argument('--samples-synthetic', type=int, default=20,help='Samples of synthetic sequences to be produced')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    filenames = cfg.DATA_LIST
    filenames = [filename.replace(".csv", ".pkl") for filename in filenames]
    filenames = [ os.path.join(cfg.PICKLE.PICKLE_DIR, filename) for filename in filenames if filename.endswith('.pkl')]
    generate_synthetic_data(cfg, filenames, args.samples_synthetic)