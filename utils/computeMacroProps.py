"""
Code to compute macroproperties data for previously aggregated data.
"""
import logging
import pickle
import pandas as pd
import numpy as np
import os
import sys
from tqdm.auto import tqdm
from myparser import getYamlConfig
from data import preProcessData, filterDataByLU, filterDataByTime, getMacroPropertiesAtTimeStamp

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("logs/ComputeMacroProps.log"),
                        logging.StreamHandler(sys.stdout)]
                    )

def computeMacroPropsATC(cfg, aggDataDir, pklDataDir, filenames, t_init=None, t_last=None):
    """
    This function computes the macroproperties for previously aggregated
    ATC data per file.

    And by providing t_init_obs and t_final, it is possible to generate sequences
    in a specific timeframe across all aggregated data files.

    Args:
        cfg: configuration file
        aggDataDir: ATC aggregated files directory.
        pklDataDir: directory to save the macroproperties data.
        filenames: list of aggregated ATC files.
        t_init_obs: init time to start the sequences, if not provided, it will be the min time of aggregated data
        t_final: final time to stop the sequences, if not provided, it will be the max time of aggregated data
    """
    seq_count = 0
    for idx, filename in enumerate(filenames):
        seq_per_file = []
        logging.info('Extracting data from: {}'.format(os.path.join(aggDataDir, filename)))
        logging.info("File {} out of {}".format(idx+1, len(filenames)))
        df = pd.read_csv(os.path.join(aggDataDir, filename), index_col=0)
        df['time'] = pd.to_datetime(df['time'])
        data, rLU = preProcessData(df, cfg=cfg, LU=cfg.MACROPROPS.LU)
        filteredData = filterDataByLU(data, cfg=cfg, LU=rLU)

        if t_init is None:
            t_init_obs = data['time'].min()
        if t_last is None:
            t_final = data['time'].max()

        t_seq = pd.to_timedelta((cfg.DATASET.PAST_LEN + cfg.DATASET.FUTURE_LEN)*cfg.CONVGRU.TIME_RES, unit='s')
        logging.info('t_init_obs + t_seq: {} and t_final {}'.format(t_init_obs + t_seq, t_final))
        while t_init_obs + t_seq <= t_final:
            t_init_current = t_init_obs
            seq = np.zeros((4, cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS, cfg.DATASET.PAST_LEN + cfg.DATASET.FUTURE_LEN))
            for obs in range(cfg.DATASET.PAST_LEN + cfg.DATASET.FUTURE_LEN):
                dataByTime = filterDataByTime(filteredData, time=t_init_obs, cfg=cfg)
                t_init_obs += pd.to_timedelta(cfg.CONVGRU.TIME_RES, unit='s')
                rho, mu_vx, mu_vy, sigma2_v = getMacroPropertiesAtTimeStamp(dataByTime, cfg, LU=rLU)
                seq[:,:,:,obs] = np.stack((rho, mu_vx, mu_vy, sigma2_v), axis=0)
            seq_per_file.append(seq)

            if cfg.MACROPROPS.OVERLAP:
                t_init_obs = t_init_current + cfg.MACROPROPS.WINDOWSIZE*cfg.MACROPROPS.TIME_RES

        seq_count += len(seq_per_file)
        seq_to_write = np.asarray(seq_per_file)
        logging.info('Total sequences in {}: {}'.format(filename, len(seq_per_file)))
        logging.info('Total memory to be written {:.4f} GB'.format(seq_to_write.nbytes / (1024*1024*1024)))
        logging.info('Total sequences so far: {}'.format(seq_count))
        # Save macroproperties per file
        try:
            with open(pklDataDir + os.path.splitext(filename)[0]+".pkl", 'wb') as file:
                pickle.dump(seq_to_write, file)
        except MemoryError:
            logging.info("MemoryError: Unable to pickle data due to memory issues.")
        except Exception as e:
            logging.info(f"An error occurred while dumping pkl file: {str(e)}")
        logging.info("-------------------------------------")

if __name__ == '__main__':
    cfg = getYamlConfig()
    filenames = os.listdir(cfg.DATA_AGGREGATION.AGG_DATA_DIR)
    filenames = [ filename for filename in filenames if filename.endswith('.csv') ]
    computeMacroPropsATC(cfg, cfg.DATA_AGGREGATION.AGG_DATA_DIR, cfg.PICKLE.PICKLE_DIR, filenames)
