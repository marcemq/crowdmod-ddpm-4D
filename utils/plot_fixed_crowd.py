"""
Code to compute macroproperties data for previously aggregated data,
and plot a fixed subset of pedrestrian crowd in order to showcase its
macroscopic properties.
"""
import argparse
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('.')
from utils.myparser import getYamlConfig
from utils.data import preProcessData, filterDataByLU, filterDataByTime, getMacroPropertiesAtTimeStamp

def computeMacroPropsATC(cfg, aggFilename, t_init):
    """
    Returns:
    - fixed trajectory crowd at t_init time
    - macroscopic properties computed for the fixed crowd
    """
    df = pd.read_csv(aggFilename, index_col=0)
    df['time'] = pd.to_datetime(df['time'])
    data, rLU = preProcessData(df, cfg=cfg, LU=cfg.MACROPROPS.LU)
    filteredData = filterDataByLU(data, cfg=cfg, LU=rLU,showMsg=True)

    macroprops = np.zeros((4, cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS))
    dataByTime = filterDataByTime(filteredData, time=t_init, cfg=cfg, showMsg=True)

    duplicated_rows1 = dataByTime.iloc[[0, 5, 15]].copy()
    duplicated_rows2 = dataByTime.iloc[[11,8,7]].copy()
    duplicated_rows1['pos_y'] -=4
    duplicated_rows2['pos_y'] +=2
    duplicated_rows2['pos_x'] +=0.8
    dataByTime = pd.concat([dataByTime, duplicated_rows1, duplicated_rows2], ignore_index=True)

    duplicated_rows = dataByTime.iloc[[10,14]].copy()
    duplicated_rows['pos_x'] +=1.5
    duplicated_rows['pos_y'] -=2
    duplicated_rows['vel_x'] *=1.5
    duplicated_rows['vel_y'] *=-1
    dataByTime = pd.concat([dataByTime, duplicated_rows], ignore_index=True)

    duplicated_rows = dataByTime.iloc[[10]].copy()
    duplicated_rows['pos_x'] +=0.9
    duplicated_rows['pos_y'] -=2.2
    duplicated_rows['vel_x'] *=0.5
    duplicated_rows['vel_y'] *=-1
    dataByTime = pd.concat([dataByTime, duplicated_rows], ignore_index=True)

    rho, mu_vx, mu_vy, sigma2_v = getMacroPropertiesAtTimeStamp(dataByTime, cfg, LU=rLU)
    macroprops[:,:,:] = np.stack((rho, mu_vx, mu_vy, sigma2_v), axis=0)

    return dataByTime, macroprops, rLU

def plotFixedTrajCrowdAndMacro(cfg, aggFilename, t_init):
    ROWS, COLS=12,12
    crowd, macroprops, rLU = computeMacroPropsATC(cfg, aggFilename, t_init)

    x, y = np.mgrid[0:COLS, 0:COLS]
    fig, ax = plt.subplots()
    axp = ax.matshow(macroprops[0,:,:], cmap=plt.cm.Blues)
    Q = ax.quiver(macroprops[1,:,:], -macroprops[2,:,:], color='green', angles='xy',scale_units='xy', scale=1,width=0.007)
    cbar=plt.colorbar(axp, cmap=plt.cm.Blues, fraction=0.017, pad=0.04)
    cbar.ax.text(2, 3.3, 'Density',va='center', ha='center', fontsize=11)

    for i in range(ROWS):
        for j in range(COLS):
            center = (x[j,i]+macroprops[1,i,j], y[j,i]-macroprops[2,i,j])
            circle = plt.Circle(center, 4*np.sqrt(macroprops[3,i,j]), fill=False, color='green', lw=1.5)
            Q.axes.add_artist(circle)

    crowd['pos_i'] = np.abs(((crowd['pos_y'].to_numpy() - (rLU[1]-0.7))/cfg.MACROPROPS.DY).reshape(-1))
    crowd['pos_j'] = ((crowd['pos_x'].to_numpy() - (rLU[0]+0.5))/cfg.MACROPROPS.DX).reshape(-1)
    ax.quiver(crowd['pos_j'], crowd['pos_i'], crowd['vel_x'], -crowd['vel_y'], color='red', angles='xy', scale_units='xy', scale=1,width=0.005)

    ax.scatter(crowd["pos_j"], crowd["pos_i"], c='r', s=10.0)
    plt.show()
    fig.savefig("images/macroPropsAndCrowd.pdf", format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to plot a pedestrian crowd and its macroscopic properties.")
    parser.add_argument('--config-yml-file', type=str, default='config/ATC_ddpm_4test.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/ATC_ddpm_DSlist4test_one.yml',help='Configuration YML macroprops list for specific dataset.')

    args = parser.parse_args()

    aggFilename = "datasets/ATC/aggData/raw30minData4UT.csv"
    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    t_init =  datetime.strptime("2012-11-14 12:26:29", '%Y-%m-%d %H:%M:%S')

    plotFixedTrajCrowdAndMacro(cfg, aggFilename, t_init)
    # How to execute: at root repository
    # python3 utils/plot_fixed_crowd.py