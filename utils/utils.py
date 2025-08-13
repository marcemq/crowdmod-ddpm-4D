import os
import wandb
import logging
from utils.dataset import getDataset, getClassicDataset

def create_directory(directory_path):
    """
    Create a directory if it does not exist.

    Args:
        directory_path (str): The path of the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Directory '{directory_path}' created successfully.")
    else:
        logging.info(f"Directory '{directory_path}' already exists.")

def get_filenames_paths(cfg):
    """
    Return list of filenames with complete path.
    """
    filenames = cfg.DATA_LIST
    if cfg.DATASET.NAME in ["ATC", "ATC4TEST"]:
        filenames = [filename.replace(".csv", ".pkl") for filename in filenames]
    elif cfg.DATASET.NAME in ["HERMES-BO", "HERMES-CR-120", "HERMES-CR-120-OBST"]:
        filenames = [filename.replace(".txt", ".pkl") for filename in filenames]
    else:
        logging.info("Dataset not supported")

    filenames = [ os.path.join(cfg.DATA_FS.PICKLE_DIR, filename) for filename in filenames if filename.endswith('.pkl')]
    return filenames

def get_training_dataset(cfg, filenames, mprops_count):
    """
    Return training and validation data for specific dataset type.
    """
    if cfg.DATASET.DATASET_TYPE == "BySplitRatio":
        batched_train_data, batched_val_data = getClassicDataset(cfg, filenames, mprops_count=mprops_count)
    elif cfg.DATASET.DATASET_TYPE == "ByFilenames":
        batched_train_data, batched_val_data, _ = getDataset(cfg, filenames, mprops_count=mprops_count)
    else:
        logging.error(f"Dataset type not supported.")
    logging.info(f"Batched Train dataset loaded.")

    return batched_train_data, batched_val_data

def get_test_dataset(cfg, filenames, mprops_count):
    """
    Return testing data for specific dataset type.
    """
    if cfg.DATASET.DATASET_TYPE == "BySplitRatio":
        _, batched_test_data = getClassicDataset(cfg, filenames, mprops_count=mprops_count)
    elif cfg.DATASET.DATASET_TYPE == "ByFilenames":
        _, _, batched_test_data = getDataset(cfg, filenames, test_data_only=True, mprops_count=mprops_count)
    else:
        logging.error(f"Dataset type not supported.")
    logging.info(f"Batched Test dataset loaded.")

    return batched_test_data

def get_checkpoint_save_path(cfg, arch, epoch):
    """
    Return checkpoint save complete path based on arch.
    """
    if arch == "DDPM-UNet":
        save_path = cfg.DATA_FS.SAVE_DIR+(cfg.MODEL.NAME.format(arch, cfg.MODEL.DDPM.TRAIN.EPOCHS, cfg.DATASET.PAST_LEN, cfg.DATASET.FUTURE_LEN, epoch, cfg.DATASET.VELOCITY_NORM))
    elif arch == "ConvGRU":
        save_path = cfg.DATA_FS.SAVE_DIR+(cfg.MODEL.NAME.format(arch, cfg.MODEL.CONVGRU.TRAIN.EPOCHS, cfg.DATASET.PAST_LEN, cfg.DATASET.FUTURE_LEN, epoch, cfg.DATASET.VELOCITY_NORM))
    else:
        logging.error("Architecture not supported.")

    return save_path

def get_model_fullname(cfg, arch, epoch):
    """
    Return model fullname based on arch.
    """
    if arch == "DDPM-UNet":
        model_fullname = cfg.DATA_FS.SAVE_DIR+(cfg.MODEL.NAME.format(arch, cfg.MODEL.DDPM.TRAIN.EPOCHS, cfg.DATASET.PAST_LEN, cfg.DATASET.FUTURE_LEN, epoch, cfg.DATASET.VELOCITY_NORM))
    elif arch == "ConvGRU":
        model_fullname = cfg.DATA_FS.SAVE_DIR+(cfg.MODEL.NAME.format(arch, cfg.MODEL.CONVGRU.TRAIN.EPOCHS, cfg.DATASET.PAST_LEN, cfg.DATASET.FUTURE_LEN, epoch, cfg.DATASET.VELOCITY_NORM))
    else:
        logging.error("Architecture not supported.")

    return model_fullname

def init_wandb(cfg, arch):
    """
    Initialize W&B based on arch
    """
    if arch == "DDPM-UNet":
        wandb.init(
            project="macroprops-predict-4D",
            config={
                "architecture": arch,
                "dataset": cfg.DATASET.NAME,
                "learning_rate": cfg.MODEL.DDPM.TRAIN.SOLVER.LR,
                "epochs": cfg.MODEL.DDPM.TRAIN.EPOCHS,
                "batch_size": cfg.DATASET.BATCH_SIZE,
                "past_len": cfg.DATASET.PAST_LEN,
                "future_len": cfg.DATASET.FUTURE_LEN,
                "weight_decay": cfg.MODEL.DDPM.TRAIN.SOLVER.WEIGHT_DECAY,
                "solver_betas": cfg.MODEL.DDPM.TRAIN.SOLVER.BETAS,
            }
        )
    elif arch == "ConvGRU":
        wandb.init(
            project="macroprops-predict-4D",
            config={
                "architecture": arch,
                "dataset": cfg.DATASET.NAME,
                "learning_rate": cfg.MODEL.CONVGRU.TRAIN.SOLVER.LR,
                "epochs": cfg.MODEL.CONVGRU.TRAIN.EPOCHS,
                "batch_size": cfg.DATASET.BATCH_SIZE,
                "past_len": cfg.DATASET.PAST_LEN,
                "future_len": cfg.DATASET.FUTURE_LEN,
                "weight_decay": cfg.MODEL.CONVGRU.TRAIN.SOLVER.WEIGHT_DECAY,
                "solver_betas": cfg.MODEL.CONVGRU.TRAIN.SOLVER.BETAS,
            }
        )
    else:
        logging.error("Architecture not supported.")