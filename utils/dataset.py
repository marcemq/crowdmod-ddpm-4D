import pickle
import random
import numpy as np
import torch
import logging
import os
from utils.data import preProcessData, filterDataByLU, filterDataByTime, getMacroPropertiesAtTimeStamp
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

class CustomTransform():
    def __call__(self, tensor):
        channels_to_normalize = 4
        stats =np.empty((4,4))
        for i in range(4):
            stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3] = np.mean(tensor[:,i,:,:]), np.std(tensor[:,i,:,:]), np.min(tensor[:,i,:,:]), np.max(tensor[:,i,:,:])
        # density and var channels won't be normalized
        for channel in range(1, channels_to_normalize-1):
            tensor[:,channel,:,:]=((tensor[:, channel, :, :] - stats[channel,2]) / (stats[channel,3] - stats[channel,2])) * 2 - 1

        return tensor, stats

class MacropropsDataset(Dataset):
    def __init__(self, seq_all, cfg, transform=None):
        self.transform = transform
        if self.transform:
            seq_all, stats = self.transform(seq_all)

        self.X = seq_all[:,:,:,:,:cfg.DATASET.OBS_LEN]
        self.X = np.squeeze(self.X, axis=-1)
        self.Y = seq_all[:,:,:,:,cfg.DATASET.OBS_LEN:cfg.DATASET.OBS_LEN+cfg.DATASET.PRED_LEN]
        self.Y = np.squeeze(self.Y, axis=-1)
        self.stats = stats

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X_seq = self.X[idx]
        Y_seq = self.Y[idx]
        # AR: I think we need to do the inverse transform here
        return X_seq, Y_seq, self.stats

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

def getMacropropsFromFilenames(filenames):
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

    stats = np.empty((4,4))
    for i in range(4):
        stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3] = np.mean(data[:,i,:,:,:]), np.std(data[:,i,:,:,:]), np.min(data[:,i,:,:,:]), np.max(data[:,i,:,:,:])
        logging.info(f'Stats per dataset channel {i} ==> mean:{stats[i, 0]:.4f}, std:{stats[i, 1]:.4f}, min:{stats[i, 2]:.4f}, max:{stats[i, 3]:.4f}')

    return data, stats

def dataHelper(cfg, filenames):
    "Compute macroprops sequences and split data by filecount defined at config file."
    if not cfg.PICKLE.USE_PICKLE:
        logging.info("Read macroproperties data to define train, validation and test sets.")

        random.shuffle(filenames)
        train_filenames = filenames[:cfg.DATASET.TRAIN_FILE_COUNT]
        val_filenames = filenames[cfg.DATASET.TRAIN_FILE_COUNT:cfg.DATASET.TRAIN_FILE_COUNT+cfg.DATASET.VAL_FILE_COUNT]
        test_filenames = filenames[cfg.DATASET.TRAIN_FILE_COUNT+cfg.DATASET.VAL_FILE_COUNT:]

        train_data, train_stats = getMacropropsFromFilenames(train_filenames)
        val_data, val_stats = getMacropropsFromFilenames(val_filenames)
        test_data, test_stats = getMacropropsFromFilenames(test_filenames)
        #saveData(train_data, val_data, test_data, cfg.PICKLE.PICKLE_DIR)
    else:
        logging.info("Unpickling data...")
        pickle_in = open(cfg.PICKLE.PICKLE_DIR+"train_data.pkl","rb")
        train_data = pickle.load(pickle_in)
        pickle_in = open(cfg.PICKLE.PICKLE_DIR+"val_data.pkl","rb")
        val_data = pickle.load(pickle_in)
        pickle_in = open(cfg.PICKLE.PICKLE_DIR+"test_data.pkl","rb")
        test_data = pickle.load(pickle_in)
        
    logging.info("In dataHelper func, shape of train_data:{}, val_data:{}, test_data:{} from files".format(train_data.shape, val_data.shape, test_data.shape))

    return train_data, val_data, test_data, train_stats, val_stats, test_stats

def getDataset(cfg, filenames):
    if 'merge_from_file' in cfg.DATASET.params:
        del cfg.DATASET.params['merge_from_file']
    if 'merge_from_dict' in cfg.DATASET.params:
        del cfg.DATASET.params['merge_from_dict']

    # Load the dataset and perform the split
    tmp_train_data, tmp_val_data, tmp_test_data, _, _, _ = dataHelper(cfg, filenames)
    # Transfor set
    custom_transform = CustomTransform()
    # Torch dataset
    train_data= MacropropsDataset(tmp_train_data, cfg, transform=custom_transform)
    val_data  = MacropropsDataset(tmp_val_data, cfg, transform=custom_transform)
    test_data = MacropropsDataset(tmp_test_data, cfg, transform=custom_transform)
    # Form batches
    batched_train_data = DataLoader(train_data, **cfg.DATASET.params)
    batched_val_data   = DataLoader(val_data, **cfg.DATASET.params)
    batched_test_data  = DataLoader(test_data, **cfg.DATASET.params)

    return batched_train_data, batched_val_data, batched_test_data