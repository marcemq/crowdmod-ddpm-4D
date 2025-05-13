import pickle
import random
import numpy as np
import torch
import logging
import os
from utils.data import preProcessData, filterDataByLU, filterDataByTime, getMacroPropertiesAtTimeStamp
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

class CustomTransform():
    def __call__(self, tensor, cfg):
        stats = np.empty((cfg.MACROPROPS.MPROPS_COUNT, 4))
        for i in range(cfg.MACROPROPS.MPROPS_COUNT):
            stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3] = np.mean(tensor[:,i,:,:]), np.std(tensor[:,i,:,:]), np.min(tensor[:,i,:,:]), np.max(tensor[:,i,:,:])
        # only velocity channels gets normalized
        for channel in [1, 2]:
            tensor[:,channel,:,:,:]=((tensor[:, channel, :, :] - stats[channel,2]) / (stats[channel,3] - stats[channel,2])) * 2 - 1

        return tensor, stats

class MacropropsDataset(Dataset):
    def __init__(self, seq_all, cfg, transform=None):
        self.transform = transform
        if self.transform:
            seq_all, stats = self.transform(seq_all, cfg)

        self.PAST = seq_all[:,:,:,:,:cfg.DATASET.PAST_LEN]
        self.FUTURE = seq_all[:,:,:,:,cfg.DATASET.PAST_LEN:cfg.DATASET.PAST_LEN+cfg.DATASET.FUTURE_LEN]
        self.stats = stats

    def __len__(self):
        return len(self.FUTURE)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        PAST_seq = self.PAST[idx]
        FUTURE_seq = self.FUTURE[idx]
        # AR: I think we need to do the inverse transform here
        return PAST_seq, FUTURE_seq, self.stats

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
    seq_per_file_list = []
    for idx, filename in enumerate(filenames):
        logging.info('Loading macro-props data from: {}'.format(filename))
        logging.info("File {} out of {}".format(idx+1, len(filenames)))
        try:
            with open(filename, "rb") as file:
                seq_per_file = pickle.load(file)
                if np.any(np.isnan(seq_per_file )):
                    logging.info(f'{filename} has NaN values')
                    raise ValueError('The loaded data contains NaN values.')
                seq_per_file_list.append(seq_per_file)
        except MemoryError:
            logging.info("MemoryError: Unable to load pickle data due to memory issues.")
        except Exception as e:
            logging.info(f"An error occurred while loading pkl file: {str(e)}")
        logging.info("-------------------------------------")
    seq_all = np.concatenate(seq_per_file_list, axis=0)
    data = np.asarray(seq_all)

    stats = np.empty((mprops_count, 4))
    for i in range(mprops_count):
        stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3] = np.mean(data[:,i,:,:,:]), np.std(data[:,i,:,:,:]), np.min(data[:,i,:,:,:]), np.max(data[:,i,:,:,:])
        logging.info(f'Stats per dataset channel {i} ==> mean:{stats[i, 0]:.4f}, std:{stats[i, 1]:.4f}, min:{stats[i, 2]:.4f}, max:{stats[i, 3]:.4f}')

    return data[:, 0:mprops_count, :, :, :], stats

def dataHelper(cfg, filenames, mprops_count, train_data_only=False, test_data_only=False):
    "Compute macroprops sequences and split data by filecount defined at config file."
    if not cfg.PICKLE.USE_PICKLE:
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
        #saveData(train_data, val_data, test_data, cfg.PICKLE.PICKLE_DIR)
    else:
        logging.info("Unpickling data...")
        pickle_in = open(cfg.PICKLE.PICKLE_DIR+"train_data.pkl","rb")
        train_data = pickle.load(pickle_in)
        pickle_in = open(cfg.PICKLE.PICKLE_DIR+"val_data.pkl","rb")
        val_data = pickle.load(pickle_in)
        pickle_in = open(cfg.PICKLE.PICKLE_DIR+"test_data.pkl","rb")
        test_data = pickle.load(pickle_in)

    if train_data_only:
        logging.info("In dataHelper func, shape of train_data:{} from files".format(train_data.shape))
    elif test_data_only:
        logging.info("In dataHelper func, shape of test_data:{} from files".format(test_data.shape))
    else:
        logging.info("In dataHelper func, shape of train_data:{}, val_data:{}, test_data:{} from files".format(train_data.shape, val_data.shape, test_data.shape))

    return train_data, val_data, test_data, train_stats, val_stats, test_stats

def getDataset(cfg, filenames, BATCH_SIZE=None, train_data_only=False, test_data_only=False):
    if 'merge_from_file' in cfg.DATASET.params:
        del cfg.DATASET.params['merge_from_file']
    if 'merge_from_dict' in cfg.DATASET.params:
        del cfg.DATASET.params['merge_from_dict']
    if BATCH_SIZE == None:
        BATCH_SIZE = cfg.DATASET.BATCH_SIZE

    # Load the dataset and perform the split
    tmp_train_data, tmp_val_data, tmp_test_data, _, _, _ = dataHelper(cfg, filenames, cfg.MACROPROPS.MPROPS_COUNT, train_data_only, test_data_only)
    # Transfor set
    custom_transform = CustomTransform()
    train_data, val_data, test_data = None, None, None
    batched_train_data, batched_val_data, batched_test_data = None, None, None

    if train_data_only:
        # Torch dataset
        train_data= MacropropsDataset(tmp_train_data, cfg, transform=custom_transform)
        # Form batches
        batched_train_data = DataLoader(train_data, batch_size=BATCH_SIZE, **cfg.DATASET.params)
    elif test_data_only:
        # Torch dataset
        test_data = MacropropsDataset(tmp_test_data, cfg, transform=custom_transform)
        # Form batches
        batched_test_data  = DataLoader(test_data, batch_size=BATCH_SIZE, **cfg.DATASET.params)
    else:
        # Torch dataset
        train_data= MacropropsDataset(tmp_train_data, cfg, transform=custom_transform)
        val_data  = MacropropsDataset(tmp_val_data, cfg, transform=custom_transform)
        test_data = MacropropsDataset(tmp_test_data, cfg, transform=custom_transform)
        # Form batches
        batched_train_data = DataLoader(train_data, batch_size=BATCH_SIZE, **cfg.DATASET.params)
        batched_val_data   = DataLoader(val_data, batch_size=BATCH_SIZE, **cfg.DATASET.params)
        batched_test_data  = DataLoader(test_data, batch_size=BATCH_SIZE, **cfg.DATASET.params)

    return batched_train_data, batched_val_data, batched_test_data

def getClassicDataset(cfg, filenames, split_ratio=0.8):
    BATCH_SIZE = cfg.DATASET.BATCH_SIZE

    # Load all sequences from all filenames
    logging.info("Loading all macroprops sequences (no file partition)...")
    all_data, _ = getMacropropsFromFilenames(filenames, cfg.MACROPROPS.MPROPS_COUNT)

    logging.info(f"Total number of sequences loaded: {len(all_data)} of shape {all_data.shape}")

    # Torch dataset
    dataset = MacropropsDataset(all_data, cfg, transform=CustomTransform())

    # Calculate split sizes
    train_len = int(split_ratio * len(dataset))
    test_len = len(dataset) - train_len

    # Shuffle & split
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    # Form DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, **cfg.DATASET.params)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, **cfg.DATASET.params)

    logging.info(f"Train split: {len(train_dataset)}, Test split: {len(test_dataset)}")

    return train_loader, test_loader