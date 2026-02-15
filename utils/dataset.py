import pickle
import random
import numpy as np
import torch
import logging
import os
from utils.data import preProcessData, filterDataByLU, filterDataByTime, getMacroPropertiesAtTimeStamp
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

class CustomTransform():
    def __call__(self, tensor, cfg, mprops_count):
        stats = np.empty((mprops_count, 4))
        for i in range(mprops_count):
            stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3] = np.mean(tensor[:,i,:,:]), np.std(tensor[:,i,:,:]), np.min(tensor[:,i,:,:]), np.max(tensor[:,i,:,:])

        if cfg.DATASET.VELOCITY_NORM:
            # only velocity channels gets normalized in [-1,1] range
            for channel in [1, 2]:
                tensor[:,channel,:,:,:]=((tensor[:, channel, :, :] - stats[channel,2]) / (stats[channel,3] - stats[channel,2])) * 2 - 1

        return tensor, stats

class MacropropsDataset(Dataset):
    def __init__(self, seq_all, cfg, mprops_count, stride=10):
        self.mprops_count = mprops_count
        self.stride = stride
        self.past_len = cfg.DATASET.PAST_LEN
        self.future_len = cfg.DATASET.FUTURE_LEN

        total_len = seq_all.shape[-1]
        window_len = self.past_len + self.future_len

        # compute all possible window start indices with stride
        self.indices = []
        for seq_idx in range(seq_all.shape[0]):  # loop over batch dimension
            for t in range(0, total_len - window_len + 1, stride):
                self.indices.append((seq_idx, t))

        self.seq_all = seq_all

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq_idx, t = self.indices[idx]
        window = self.seq_all[seq_idx, :, :, :, t:t+self.past_len+self.future_len]

        past_seq = window[:, :, :, :self.past_len]
        future_seq = window[:, :, :, self.past_len:]

        return past_seq, future_seq

def saveData(train_data, val_data, test_data, pickle_dir):
    logging.info("Saving training, validatio and testing dara ndarrays in pickle files...")
    # Training dataset
    pickle_out = open(pickle_dir+"train_data.pkl","wb")
    pickle.dump(train_data, pickle_out, protocol=2)
    pickle_out.close()

    # Validation dataset
    pickle_out = open(pickle_dir+"val_data.pkl","wb")
    pickle.dump(val_data, pickle_out, protocol=2)
    pickle_out.close()

    # Test dataset
    pickle_out = open(pickle_dir+"test_data.pkl","wb")
    pickle.dump(test_data, pickle_out, protocol=2)
    pickle_out.close()

def getMacropropsFromFilenames(filenames_and_numSamples, mprops_count, per_sample_shape):
    total_samples = sum(num_samples for _, num_samples in filenames_and_numSamples)
    logging.info(f"Total raw samples to load: {total_samples}")

    # prepend total_samples to shape
    data_shape = (total_samples, *per_sample_shape)
    data = np.empty(data_shape, dtype=np.float32)

    current_index = 0
    for idx, (filename, num_samples) in enumerate(filenames_and_numSamples):
        logging.info(f"Loading macro-props data from: {filename}")
        logging.info(f"File {idx+1} out of {len(filenames_and_numSamples)}")

        try:
            with open(filename, "rb") as file:
                seq_per_file = pickle.load(file)
                # insert into pre-allocated tensor
                end_index = current_index + num_samples
                data[current_index:end_index, ...] = seq_per_file
                current_index = end_index

        except MemoryError:
            logging.error("MemoryError: Unable to load pickle data due to memory issues.")
        except Exception as e:
            logging.error(f"An error occurred while loading {filename}: {str(e)}")

        logging.info("-------------------------------------")

    stats = np.empty((mprops_count, 4))
    for i in range(mprops_count):
        channel_data = data[:, i, :, :, :]
        stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3] = np.mean(channel_data), np.std(channel_data), np.min(channel_data), np.max(channel_data)
        logging.info(f"Stats per dataset channel {i} ==> mean:{stats[i, 0]:.4f}, std:{stats[i, 1]:.4f}, min:{stats[i, 2]:.4f}, max:{stats[i, 3]:.4f}")

    return data[:, 0:mprops_count, :, :, :], stats

def dataHelper(cfg, filenames_and_numSamples, mprops_count, train_data_only=False, test_data_only=False):
    "Load macroprops sequences and split data by filecount defined at config file."
    if not cfg.DATA_FS.USE_PICKLE:
        logging.info("Read macroproperties data to define train, validation and test sets.")

        random.shuffle(filenames_and_numSamples)
        train_filenames_and_numSamples = filenames_and_numSamples[:cfg.DATASET.TRAIN_FILE_COUNT]
        val_filenames_and_numSamples = filenames_and_numSamples[cfg.DATASET.TRAIN_FILE_COUNT:cfg.DATASET.TRAIN_FILE_COUNT+cfg.DATASET.VAL_FILE_COUNT]
        test_filenames_and_numSamples = filenames_and_numSamples[cfg.DATASET.TRAIN_FILE_COUNT+cfg.DATASET.VAL_FILE_COUNT:cfg.DATASET.TRAIN_FILE_COUNT+cfg.DATASET.VAL_FILE_COUNT+cfg.DATASET.TEST_FILE_COUNT]

        train_data, val_data, test_data, train_stats, val_stats, test_stats = None, None, None, None, None, None
        per_sample_shape = [4, cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS, cfg.DATASET.RAW_SEQ_LEN]

        if train_data_only:
            train_data, train_stats = getMacropropsFromFilenames(train_filenames_and_numSamples, mprops_count, per_sample_shape)
        elif test_data_only:
            test_data, test_stats = getMacropropsFromFilenames(test_filenames_and_numSamples, mprops_count, per_sample_shape)
        else:
            train_data, train_stats = getMacropropsFromFilenames(train_filenames_and_numSamples, mprops_count, per_sample_shape)
            val_data, val_stats = getMacropropsFromFilenames(val_filenames_and_numSamples, mprops_count, per_sample_shape)
            test_data, test_stats = getMacropropsFromFilenames(test_filenames_and_numSamples, mprops_count, per_sample_shape)

    else:
        logging.info("Unpickling data...")
        pickle_in = open(cfg.DATA_FS.PICKLE_DIR+"train_data.pkl","rb")
        train_data = pickle.load(pickle_in)
        pickle_in = open(cfg.DATA_FS.PICKLE_DIR+"val_data.pkl","rb")
        val_data = pickle.load(pickle_in)
        pickle_in = open(cfg.DATA_FS.PICKLE_DIR+"test_data.pkl","rb")
        test_data = pickle.load(pickle_in)

    if train_data_only:
        logging.info("In dataHelper func, shape of train_data:{} from files".format(train_data.shape))
    elif test_data_only:
        logging.info("In dataHelper func, shape of test_data:{} from files".format(test_data.shape))
    else:
        logging.info("In dataHelper func, shape of train_data:{}, val_data:{}, test_data:{} from files".format(train_data.shape, val_data.shape, test_data.shape))

    return train_data, val_data, test_data, train_stats, val_stats, test_stats

def getDataset(cfg, filenames_and_numSamples, batch_size=None, train_data_only=False, test_data_only=False, mprops_count=4):
    if 'merge_from_file' in cfg.DATASET.params:
        del cfg.DATASET.params['merge_from_file']
    if 'merge_from_dict' in cfg.DATASET.params:
        del cfg.DATASET.params['merge_from_dict']
    if batch_size == None:
        batch_size = cfg.DATASET.BATCH_SIZE

    # Load the dataset and perform the split
    tmp_train_data, tmp_val_data, tmp_test_data, _, _, _ = dataHelper(cfg, filenames_and_numSamples, mprops_count, train_data_only, test_data_only)
    # Transfor set
    train_data, val_data, test_data = None, None, None
    batched_train_data, batched_val_data, batched_test_data = None, None, None

    if train_data_only:
        # Torch dataset
        train_data = MacropropsDataset(tmp_train_data, cfg, mprops_count, stride=cfg.MACROPROPS.STRIDE)
        logging.info(f"In getDataset func, total seqs in train_data:{len(train_data)}")
        # Form batches
        batched_train_data = DataLoader(train_data, batch_size=batch_size, **cfg.DATASET.params)
    elif test_data_only:
        # Torch dataset
        test_data = MacropropsDataset(tmp_test_data, cfg, mprops_count, stride=cfg.MACROPROPS.STRIDE)
        logging.info(f"In getDataset func, total seqs in test_data:{len(test_data)}")
        # Form batches
        batched_test_data  = DataLoader(test_data, batch_size=batch_size, **cfg.DATASET.params)
    else:
        # Torch dataset
        train_data= MacropropsDataset(tmp_train_data, cfg, mprops_count, stride=cfg.MACROPROPS.STRIDE)
        val_data  = MacropropsDataset(tmp_val_data, cfg, mprops_count, stride=cfg.MACROPROPS.STRIDE)
        test_data = MacropropsDataset(tmp_test_data, cfg, mprops_count, stride=cfg.MACROPROPS.STRIDE)
        logging.info(f"In getDataset func, total seqs in train_data:{len(train_data)}, val_data:{len(val_data)}, test_data:{len(test_data)}")
        # Form batches
        batched_train_data = DataLoader(train_data, batch_size=batch_size, **cfg.DATASET.params)
        batched_val_data   = DataLoader(val_data, batch_size=batch_size, **cfg.DATASET.params)
        batched_test_data  = DataLoader(test_data, batch_size=batch_size, **cfg.DATASET.params)

    return batched_train_data, batched_val_data, batched_test_data

def getDataset4Test(cfg, filenames, batch_size=None, mprops_count=4):
    if batch_size == None:
        batch_size = cfg.DATASET.BATCH_SIZE

    # Load all sequences from all filenames
    logging.info("Loading all macroprops sequences (no file partition)...")
    all_data, _ = getMacropropsFromFilenames(filenames, mprops_count)
    logging.info(f"Total number of sequences loaded for test: {len(all_data)} of shape {all_data.shape}")

    # Torch dataset
    test_dataset = MacropropsDataset(all_data, cfg, stride=cfg.MACROPROPS.STRIDE)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, **cfg.DATASET.params)

    logging.info(f"Test sequences: {len(test_dataset)}")

    return test_loader

def getClassicDataset(cfg, filenames_and_numSamples, batch_size=None, split_ratio=0.9, mprops_count=4):
    if batch_size == None:
        batch_size = cfg.DATASET.BATCH_SIZE

    # Load all sequences from all filenames
    logging.info("Loading all macroprops sequences (no file partition)...")
    per_sample_shape = [4, cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS, cfg.DATASET.RAW_SEQ_LEN]
    all_data, _ = getMacropropsFromFilenames(filenames_and_numSamples, mprops_count, per_sample_shape)

    logging.info(f"Total number of sequences loaded: {len(all_data)} of shape {all_data.shape}")
    # ---- GLOBAL NORMALIZATION (quick experiment) ----
    mean = all_data.mean(axis=(0, 2, 3, 4))
    std  = all_data.std(axis=(0, 2, 3, 4))
    std  = np.clip(std, 1e-6, None)

    all_data = (all_data - mean[None, :, None, None, None]) / std[None, :, None, None, None]
    logging.info(f"Applied global normalization")
    # --------------------------------------------------
    # Torch dataset: complete dataset and stride applied
    dataset = MacropropsDataset(all_data, cfg, mprops_count, stride=cfg.MACROPROPS.STRIDE)
    logging.info(f"Total number of sequences in dataset: {len(dataset)}")
    # Calculate split sizes
    train_len = int(split_ratio * len(dataset))
    test_len = len(dataset) - train_len

    # Shuffle & split
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    # Form DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, **cfg.DATASET.params)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, **cfg.DATASET.params)

    logging.info(f"Train split: {len(train_dataset)}, Test split: {len(test_dataset)}")

    return train_loader, test_loader