import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from myparser import getYamlConfig
#function that maps frame_ID to unixtime
#The time difference between any two consecutive frames is 0.4
def unixtime(df, init_time = 1694563200.0):
    df['time'] = df['time'].astype(float)
    for i in range(df.shape[0]):
        df.loc[i, 'time'] = init_time + df['time'][i]*0.4/10

    return df    
#function to calculate the motion angle
def angle(x1, y1, x2, y2):
  delta_x = x2 - x1
  delta_y = y2 - y1

  return np.arctan2(delta_y, delta_x)
#function to calculate the distance given two points
def distance(x1, y1, x2, y2):
  return np.sqrt((x1-x2)**2 + (y1-y2)**2)

#function that modifies dataframe given a left upper point 
#and grid dimensions
def newDF_LU(df, LU, grid1, grid2):
    df = df[(df['pos_x'] > LU[0]) & 
    (df['pos_x'] < LU[0] + grid1) & 
    (df['pos_y'] > LU[1] - grid2) &
    (df['pos_y'] < LU[1])].reset_index(drop=True)

    df['pos_x'] -= LU[0]
    df['pos_y'] -= (LU[1]-grid2)

    return df




#-------------------------------------------------------------------------------------------
cfg = getYamlConfig('config/ETH_ddpm.yml')

#path that contains txt files
path = cfg.DATA_AGGREGATION.TXT_DATA_DIR
files = glob.glob(os.path.join(path, '*')) #extract files path
# Define the column names (headers)
column_names = ["time", "agent_ID", "pos_x", "pos_y"]

# Read the .txt file into a DataFrame
for input_file in files:
    df = pd.read_csv(input_file, delimiter="\t", header=None, names=column_names)
    df = unixtime(df)
    if input_file == 'txtFiles/biwi_hotel.txt': #we realize a rotation 
        df['pos_x'], df['pos_y'] = -df['pos_y'], df['pos_x']

    n = len(path)
    df.to_csv(cfg.DATA_AGGREGATION.AGG_DATA_DIR + input_file[n:-4] + '.csv', index=False)



path = cfg.DATA_AGGREGATION.AGG_DATA_DIR
files = glob.glob(os.path.join(path, '*')) #extract files path


#we specify the amount of cols and rows
grid1 = cfg.MACROPROPS.COLS
grid2 = cfg.MACROPROPS.ROWS
#for every file we find the left upper point that maximizes the density
for input_file in files:
    df = pd.read_csv(input_file)
    minX, minY, maxX, maxY = int(df['pos_x'].min())+1, int(df['pos_y'].min())+1, int(df['pos_x'].max()), int(df['pos_y'].max())
    
    iMax, jMax = 0, grid2
    maxCount = 0
    for i in range(minX, maxX - grid1 + 1):
        for j in range(minY + grid2, maxY + 1):
            df_aux = newDF_LU(df, [i, j], grid1, grid2)
            if maxCount < df_aux.shape[0]:
                maxCount = df_aux.shape[0]
                iMax, jMax = i, j


    df = newDF_LU(df, [iMax, jMax], grid1, grid2)  
    df.to_csv(input_file, index=False)     


#for every file we add two columns: velocity and motion_angle
for path_file in files:
    df = pd.read_csv(path_file)
    #we transform the position in m to mm
    df['pos_x'] *= 1000
    df['pos_y'] *= 1000

    df['vel'] = df['pos_x']
    df['motion_angle'] = df['pos_x']

    A = [[] for _ in range(int(df['agent_ID'].max() + 1))]

    for i in range(1, len(A)):
        A[i] = df.loc[df['agent_ID'] == i].index

        if(len(A[i]) == 1): 
            df = df.loc[df['agent_ID'] != i]
            continue
        
        for j in range(1, A[i].shape[0]):
            dis = distance(df.loc[A[i][j-1], 'pos_x'], df.loc[A[i][j-1], 'pos_y'],
                               df.loc[A[i][j], 'pos_x'], df.loc[A[i][j], 'pos_y'])
                
            df.loc[A[i][j], 'motion_angle'] = angle(df.loc[A[i][j-1], 'pos_x'], df.loc[A[i][j-1], 'pos_y'],
                                                    df.loc[A[i][j], 'pos_x'], df.loc[A[i][j], 'pos_y'])
                
            df.loc[A[i][j], 'vel'] = dis / (df.loc[A[i][j], 'time']-df.loc[A[i][j-1], 'time'])

        if(len(A[i]) > 0):
            df.loc[A[i][0], 'vel'] = df.loc[A[i][1], 'vel']
            df.loc[A[i][0], 'motion_angle'] = df.loc[A[i][1], 'motion_angle']     

    df.to_csv(path_file, index=False)