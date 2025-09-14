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
    def __init__(self, seq_all, cfg, mprops_count, transform=None, stride=10):
        self.transform = transform
        self.mprops_count = mprops_count
        self.stride = stride
        self.past_len = cfg.DATASET.PAST_LEN
        self.future_len = cfg.DATASET.FUTURE_LEN

        if self.transform:
            seq_all, stats = self.transform(seq_all, cfg, mprops_count)
        else:
            stats = None

        total_len = seq_all.shape[-1]
        window_len = self.past_len + self.future_len

        # compute all possible window start indices with stride
        self.indices = []
        for seq_idx in range(seq_all.shape[0]):  # loop over batch dimension
            for t in range(0, total_len - window_len + 1, stride):
                self.indices.append((seq_idx, t))

        self.seq_all = seq_all
        self.stats = stats

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq_idx, t = self.indices[idx]
        window = self.seq_all[seq_idx, :, :, :, t:t+self.past_len+self.future_len]

        past_seq = window[:, :, :, :self.past_len]
        future_seq = window[:, :, :, self.past_len:]

        return past_seq, future_seq, self.stats

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

def getMacropropsFromFilenames(filenames, mprops_count):
    logging.info(f"Loading {len(filenames)} macro-props files...")

    # Concatenate incrementally to avoid double memory usage
    data = None
    for idx, filename in enumerate(filenames):
        logging.info(f"Loading macro-props data from: {filename}")
        logging.info(f"File {idx+1} out of {len(filenames)}")

        try:
            with open(filename, "rb") as file:
                seq_per_file = pickle.load(file)

                if data is None:
                    data = seq_per_file
                else:
                    data = np.concatenate((data, seq_per_file), axis=0)

        except MemoryError:
            logging.error("MemoryError: Unable to load pickle data due to memory issues.")
            break
        except Exception as e:
            logging.error(f"Error loading {filename}: {str(e)}")
        logging.info("-------------------------------------")

    if data is None:
        raise RuntimeError("No data loaded.")

    # Preallocate stats
    stats = np.empty((mprops_count, 4))
    for i in range(mprops_count):
        channel_data = data[:, i, :, :, :]
        stats[i] = [channel_data.mean(), channel_data.std(), channel_data.min(), channel_data.max()]
        logging.info(f"Stats per dataset channel {i} ==>: mean={stats[i,0]:.4f}, std={stats[i,1]:.4f}, "f"min={stats[i,2]:.4f}, max={stats[i,3]:.4f}")

    return data[:, 0:mprops_count, :, :, :], stats

def dataHelper(cfg, filenames, mprops_count, train_data_only=False, test_data_only=False):
    "Load macroprops sequences and split data by filecount defined at config file."
    if not cfg.DATA_FS.USE_PICKLE:
        logging.info("Read macroproperties data to define train, validation and test sets.")

        random.shuffle(filenames)
        train_filenames = filenames[:cfg.DATASET.TRAIN_FILE_COUNT]
        val_filenames = filenames[cfg.DATASET.TRAIN_FILE_COUNT:cfg.DATASET.TRAIN_FILE_COUNT+cfg.DATASET.VAL_FILE_COUNT]
        test_filenames = filenames[cfg.DATASET.TRAIN_FILE_COUNT+cfg.DATASET.VAL_FILE_COUNT:cfg.DATASET.TRAIN_FILE_COUNT+cfg.DATASET.VAL_FILE_COUNT+cfg.DATASET.TEST_FILE_COUNT]

        train_data, val_data, test_data, train_stats, val_stats, test_stats = None, None, None, None, None, None

        if train_data_only:
            train_data, train_stats = getMacropropsFromFilenames(train_filenames, mprops_count)
        elif test_data_only:
            test_data, test_stats = getMacropropsFromFilenames(test_filenames, mprops_count)
        else:
            train_data, train_stats = getMacropropsFromFilenames(train_filenames, mprops_count)
            val_data, val_stats = getMacropropsFromFilenames(val_filenames, mprops_count)
            test_data, test_stats = getMacropropsFromFilenames(test_filenames, mprops_count)

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

def getDataset(cfg, filenames, batch_size=None, train_data_only=False, test_data_only=False, mprops_count=4):
    if 'merge_from_file' in cfg.DATASET.params:
        del cfg.DATASET.params['merge_from_file']
    if 'merge_from_dict' in cfg.DATASET.params:
        del cfg.DATASET.params['merge_from_dict']
    if batch_size == None:
        batch_size = cfg.DATASET.BATCH_SIZE

    # Load the dataset and perform the split
    tmp_train_data, tmp_val_data, tmp_test_data, _, _, _ = dataHelper(cfg, filenames, mprops_count, train_data_only, test_data_only)
    # Transfor set
    custom_transform = CustomTransform()
    train_data, val_data, test_data = None, None, None
    batched_train_data, batched_val_data, batched_test_data = None, None, None

    if train_data_only:
        # Torch dataset
        train_data = MacropropsDataset(tmp_train_data, cfg, mprops_count, transform=None, stride=cfg.MACROPROPS.STRIDE)
        logging.info(f"In getDataset func, total seqs in train_data:{len(train_data)}")
        # Form batches
        batched_train_data = DataLoader(train_data, batch_size=batch_size, **cfg.DATASET.params)
    elif test_data_only:
        # Torch dataset
        test_data = MacropropsDataset(tmp_test_data, cfg, mprops_count, transform=None, stride=cfg.MACROPROPS.STRIDE)
        logging.info(f"In getDataset func, total seqs in test_data:{len(test_data)}")
        # Form batches
        batched_test_data  = DataLoader(test_data, batch_size=batch_size, **cfg.DATASET.params)
    else:
        # Torch dataset
        train_data= MacropropsDataset(tmp_train_data, cfg, mprops_count, transform=None, stride=cfg.MACROPROPS.STRIDE)
        val_data  = MacropropsDataset(tmp_val_data, cfg, mprops_count, transform=None, stride=cfg.MACROPROPS.STRIDE)
        test_data = MacropropsDataset(tmp_test_data, cfg, mprops_count, transform=None, stride=cfg.MACROPROPS.STRIDE)
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
    test_dataset = MacropropsDataset(all_data, cfg, transform=CustomTransform(), stride=cfg.MACROPROPS.STRIDE)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, **cfg.DATASET.params)

    logging.info(f"Test sequences: {len(test_dataset)}")

    return test_loader

def getClassicDataset(cfg, filenames, batch_size=None, split_ratio=0.9, mprops_count=4):
    if batch_size == None:
        batch_size = cfg.DATASET.BATCH_SIZE

    # Load all sequences from all filenames
    logging.info("Loading all macroprops sequences (no file partition)...")
    all_data, _ = getMacropropsFromFilenames(filenames, mprops_count)

    logging.info(f"Total number of sequences loaded: {len(all_data)} of shape {all_data.shape}")

    # Torch dataset
    dataset = MacropropsDataset(all_data, cfg, mprops_count, transform=CustomTransform(), stride=cfg.MACROPROPS.STRIDE)
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