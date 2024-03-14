"""
Code for aggregate pedestrians in crowd datasets by TIME_RES=0.5 secs.
"""
import logging
import pandas as pd
import os

from tqdm.auto import tqdm
from myparser import getYamlConfig

logging.basicConfig(filename='logs/Aggregation.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

def aggregateATCData(colNames, readColNames, inDataDir, outDataDir, filenames):
    """
    This function aggregates time, pos_x, pos_y, vel, motion_angle
    by TIME_RES period of time for each pedestrian for each ATC file.

    Args:
        colNames: Columns names of ATC file csv data.
        readColNames: Columns names that are going to be read.
        inDataDir: ATC files directory.
        outDataDir: directory to save the aggregated data.
        filenames: list of ATC files.

    """
    pbar = tqdm(filenames)
    for index, filename in enumerate(filenames):
        pbar.set_description("Aggregation data for: {} (file {})".format(filename,index+1), refresh=True)
        df = pd.read_csv(inDataDir + filename, names=colNames, header=None, usecols=readColNames)

        logging.info("Aggregation data in: {}".format(filename))
        logging.info("File {} out of {}".format(index+1, len(filenames)))
        logging.info("Total samples before aggregation:{}".format(df.shape[0]))

        # compute time column as human-readable
        df['time'] = pd.to_datetime(df['time'], unit='s') + pd.to_timedelta(9, unit='hours')
        df = df.sort_values(by = 'time')
        # compute data aggregation by o.5 sec
        finalDF = df.groupby([pd.Grouper(key='time', freq='500ms'), 'personID'], as_index=False).mean()
        finalDF.to_csv(outDataDir + filename)
        
        logging.info("Total samples after aggregation:{}".format(finalDF.shape[0]))
        logging.info("-------------------------------------")
        pbar.update()

if __name__ == '__main__':
    colNames =['time', 'personID', 'pos_x', 'pos_y', 'pos_z', 'vel', 'motion_angle', 'facing_angle']
    readColNames =['time', 'personID', 'pos_x', 'pos_y', 'vel', 'motion_angle']
    cfg = getYamlConfig()

    filenames = cfg.SUNDAY_DATA_LIST
    
    aggregateATCData(colNames, readColNames, cfg.DATA_AGGREGATION.RAW_DATA_DIR, cfg.DATA_AGGREGATION.AGG_DATA_DIR, filenames)