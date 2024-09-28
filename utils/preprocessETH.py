import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from myparser import getYamlConfig
def unixtime(df, init_time = 1694563200.0):
    #function that maps frame_ID to unixtime
    df['time'] = df['time'].astype(float)
    for i in range(df.shape[0]):
        df.loc[i, 'time'] = init_time + df['time'][i]*0.4/10

    return df    

def angle(x1, y1, x2, y2):
    #function to calculate the motion angle
    delta_x = x2 - x1
    delta_y = y2 - y1

    return np.arctan2(delta_y, delta_x)

def distance(x1, y1, x2, y2):
    #function to calculate the distance given two points
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)
 
def newDF_LU(df, LU, cols, rows):
    #function that modifies dataframe given a left upper point
    df = df[(df['pos_x'] > LU[0]) & 
    (df['pos_x'] < LU[0] + cols) & 
    (df['pos_y'] > LU[1] - rows) &
    (df['pos_y'] < LU[1])].reset_index(drop=True)

    df['pos_x'] -= LU[0]
    df['pos_y'] -= (LU[1]-rows)

    return df

def generate_csv(raw_path, agg_path):
    files = glob.glob(os.path.join(raw_path, '*'))
    # Define the column names (headers)
    column_names = ["time", "agent_ID", "pos_x", "pos_y"]

    # Read the .txt file into a DataFrame
    for input_file in files:
        df = pd.read_csv(input_file, delimiter="\t", header=None, names=column_names)
        df = unixtime(df)
        if input_file == 'txtFiles/biwi_hotel.txt': #we realize a rotation 
            df['pos_x'], df['pos_y'] = -df['pos_y'], df['pos_x']

        n = len(raw_path)
        df.to_csv(agg_path + input_file[n:-4] + '.csv', index=False)

def find_LU(cfg, agg_path):
    files = glob.glob(os.path.join(agg_path, '*')) #extract files path  
    #we specify the amount of cols and rows
    cols = cfg.MACROPROPS.COLS
    rows = cfg.MACROPROPS.ROWS
    #for every file we find the left upper point that maximizes the density
    for input_file in files:
        df = pd.read_csv(input_file)
        minX, minY, maxX, maxY = int(df['pos_x'].min())+1, int(df['pos_y'].min())+1, int(df['pos_x'].max()), int(df['pos_y'].max())
        iMax, jMax = 0, rows
        maxCount = 0
        for i in range(minX, maxX - cols + 1):
            for j in range(minY + rows, maxY + 1):
                df_aux = newDF_LU(df, [i, j], cols, rows)
                if maxCount < df_aux.shape[0]:
                    maxCount = df_aux.shape[0]
                    iMax, jMax = i, j


        df = newDF_LU(df, [iMax, jMax], cols, rows)  
        df.to_csv(input_file, index=False) 

def add_vel_angle(agg_path):
    files = glob.glob(os.path.join(agg_path, '*'))
    #for every file we add two columns: velocity and motion_angle
    for path_file in files:
        df = pd.read_csv(path_file)
        #we transform the position in m to mm
        df['pos_x'] *= 1000
        df['pos_y'] *= 1000

        df['vel'] = df['pos_x']
        df['motion_angle'] = df['pos_x']

        indexes_per_agent = [[] for _ in rBnge(int(df['agent_ID'].max() + 1))]

        for i in range(1, len(indexes_per_agent)):
            indexes_per_agent[i] = df.loc[df['agent_ID'] == i].index

            if(len(indexes_per_agent[i]) == 1): 
                df = df.loc[df['agent_ID'] != i]
                continue
            
            for j in range(1, indexes_per_agent[i].shape[0]):
                dis = distance(df.loc[indexes_per_agent[i][j-1], 'pos_x'], df.loc[indexes_per_agent[i][j-1], 'pos_y'],
                                df.loc[indexes_per_agent[i][j], 'pos_x'], df.loc[indexes_per_agent[i][j], 'pos_y'])
                    
                df.loc[indexes_per_agent[i][j], 'motion_angle'] = angle(df.loc[indexes_per_agent[i][j-1], 'pos_x'], df.loc[indexes_per_agent[i][j-1], 'pos_y'],
                                            df.loc[indexes_per_agent[i][j], 'pos_x'], df.loc[indexes_per_agent[i][j], 'pos_y'])
                    
                df.loc[indexes_per_agent[i][j], 'vel'] = dis / (df.loc[indexes_per_agent[i][j], 'time']-df.loc[indexes_per_agent[i][j-1], 'time'])

            if(len(indexes_per_agent[i]) > 0):
                df.loc[indexes_per_agent[i][0], 'vel'] = df.loc[indexes_per_agent[i][1], 'vel']
                df.loc[indexes_per_agent[i][0], 'motion_angle'] = df.loc[indexes_per_agent[i][1], 'motion_angle']     

        df.to_csv(path_file, index=False)        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to preprocess ETH data.")
    parser.add_argument('--config-yml-file', type=str, default='config/ETH_ddpm.yml', help='Configuration YML file for specific dataset.')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file)
    generate_csv(cfg.DATA_AGGREGATION.RAW_DATA_DIR, cfg.DATA_AGGREGATION.AGG_DATA_DIR)
    find_LU(cfg, cfg.DATA_AGGREGATION.AGG_DATA_DIR)
    add_vel_angle(cfg.DATA_AGGREGATION.AGG_DATA_DIR)
