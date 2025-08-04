from datetime import datetime
import numpy as np
import pandas as pd
from utils.myparser import getYamlConfig
from utils.plot.plot import plotPeopleDensity, plotDataAndItsRotation, drawMacroProps
from utils.crowd import Crowd
pd.options.mode.chained_assignment = None  # default='warn'

T_INIT = datetime.strptime('2012-11-14 12:15:00', '%Y-%m-%d %H:%M:%S')  #'2012-10-24 17:00:20' 10 min, 15, 18, 28:15p
T_FINAL = '2012-10-24 17:01:00'

def getIndex(pos_x, pos_y, cfg, LU):
    """
    This function returns i,j position in grid given Left and Upper bounds.

    Args:
        pos_x: 1D array of position in x axis given in meters.
        pos_y: 1D array of position in y axis given in meters.
        LU: [L,U] left and upper bounds in meters.

    Returns:
        - 1D array corresponding to i-th index.
        - 1D array corresponding to j-th index.
    """
    i = np.abs(np.floor((pos_y-(LU[1]-1))/cfg.MACROPROPS.DY).astype(int).reshape(-1))
    j = np.floor((pos_x-LU[0])/cfg.MACROPROPS.DX).astype(int).reshape(-1)
    return i, j

def computeMacroPropsInROI(filename, lu, t_init, cfg, frames=10):
    """
    This function returns macro properties time serie of lenght 'frames' in grid of interest.

    Args:
        filename: of an specific file part of ATC dataset.
        LU: [L,U] left and upper bounds in meters.
        t_init: datetime object in '%Y-%m-%d %H:%M:%S' format.
        frames: length of time serie.

    Returns:
        - 4D array of size (ROWS, COLS, 4, frames)
    """
    macroProps = np.zeros((cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS, 4, frames))
    df = pd.read_csv(filename, index_col=0)
    data = preProcessData(df)
    filteredData = filterDataByLU(data, lu)

    for frame in range(frames):
        dataByTime = filterDataByTime(filteredData, time=t_init)
        t_init += pd.to_timedelta(cfg.CONVGRU.TIME_RES, unit='s')

        rho, mu_vx, mu_vy, sigma2_v = getMacroPropertiesAtTimeStamp(dataByTime, LU=lu)
        macroProps[:,:,:,frame] = np.stack((rho, mu_vx, mu_vy, sigma2_v), axis=2)

    return macroProps

def getMacroPropertiesAtTimeStamp(data, cfg, LU):
    """
    This function returns macro properties of given dataframe in a specific timestamp
    and grid of interest.

    Args:
        data: dataframe within specific timestamp+TIME_RES and grid of interest.
        LU: [L,U] left and upper bounds in meters.

    Returns:
        - rho: 2D array of density.
        - mu_vx: 2D array of averaged velocity in x axis.
        - mu_vy: 2D array of averaged velocity in y axis.
        - sigma2_v: 2D array of variance of norm velocity
    """
    i, j = getIndex(data['pos_x'].to_numpy(), data['pos_y'].to_numpy(), cfg, LU)
    data['i'], data['j'] = i, j

    rho = np.zeros((cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS))
    mu_vx = np.zeros((cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS))
    mu_vy = np.zeros((cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS))
    exp_vel_norm = np.zeros((cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS))
    sigma2_v = np.zeros((cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS))
    ndarrayData=data.to_numpy()
    
    for tmp in range(i.shape[0]):
        rho[i[tmp], j[tmp]] += 1
        mu_vx[i[tmp], j[tmp]] += ndarrayData[tmp,6]
        mu_vy[i[tmp], j[tmp]] += ndarrayData[tmp,7]
        exp_vel_norm[i[tmp], j[tmp]] += ndarrayData[tmp,8]

    mu_vx = mu_vx/(rho+cfg.MACROPROPS.EPS)
    mu_vy = mu_vy/(rho+cfg.MACROPROPS.EPS)
    exp_vel_norm = exp_vel_norm/(rho+cfg.MACROPROPS.EPS)

    for tmp in range(i.shape[0]):
        sigma2_v[i[tmp], j[tmp]] += (ndarrayData[tmp,8] - exp_vel_norm[i[tmp], j[tmp]])**2

    sigma2_v = sigma2_v/(rho+cfg.MACROPROPS.EPS)

    return rho, mu_vx, mu_vy, sigma2_v

def filterDataByTime(data, time, cfg, showMsg=False):
    """
    This function returns a dataframe filtered by time + TIME_RES.

    Args:
        data: dataframe read from a ATC dataset.
        time: datetime object in '%Y-%m-%d %H:%M:%S' format.
        showMsg: flag to print a message in console.

    Returns:
        - The dataframe filtered by time + TIME_RES.
    """
    data['time'] = pd.to_datetime(data['time'])
    data = data[(data['time'] >= time) & (data['time']<time + pd.to_timedelta(cfg.MACROPROPS.TIME_RES, unit='s'))]
    if showMsg:
        print("Pedestrians in timestamp: {}".format(data.shape[0]))
    return data

def filterDataByLU(data, cfg, LU, showMsg=False):
    """
    This function returns a dataframe filtered by Left and Upper bounds upto
    ROWS and COLS corresponding to grid of interest.

    Args:
        data: dataframe read from a ATC dataset.
        LU: [L,U] left and upper bounds in meters.
        showMsg: flag to print a message in console.

    Returns:
        - The dataframe within grid of interest.
    """
    data = data[(data['pos_x']>=LU[0]) & (data['pos_x']<LU[0]+(cfg.MACROPROPS.COLS*cfg.MACROPROPS.DX))]
    data = data[(data['pos_y']<=LU[1]) & (data['pos_y']>LU[1]-(cfg.MACROPROPS.ROWS*cfg.MACROPROPS.DY))]
    if showMsg:
        print("Pedestrians in grid:{}".format(data.shape[0]))
    return data

def preProcessData(df, cfg, LU):
    """
    This function returns a dataframe were pos_x, pos_y, vel_x, vel_y and vel_norm are
    computed in m/s and properly rotated.

    Args:
        df: dataframe read from a ATC dataset.
        LU: [L,U] left and upper bounds in meters

    Returns:
        - The dataframe with its columns computed in m/s and rotated by R
        - a rotated version of LU for further use
    """
    # Angle for data rotation
    if cfg.DATASET.NAME == "ETH-UCY-4D":
        THETA = cfg.MACROPROPS.THETA
    elif cfg.DATASET.NAME in ["ATC-4D", "ATC4TEST-4D"]:
        THETA = np.pi-cfg.MACROPROPS.THETA
    R = np.array([[np.cos(THETA), -np.sin(THETA)], [np.sin(THETA), np.cos(THETA)]])
    # compute XY positions in meters
    df['pos_x'] = df['pos_x']/1000.0
    df['pos_y'] = df['pos_y']/1000.0
    rotated_data = R@np.stack((df['pos_x'], df['pos_y']), axis=0)
    df['pos_x'] = rotated_data[0,:]
    df['pos_y'] = rotated_data[1,:]
    # compute vel_x, vel_y
    df['vel_x'] = df['vel']/1000.0*np.cos(df['motion_angle'] + THETA)
    df['vel_y'] = df['vel']/1000.0*np.sin(df['motion_angle'] + THETA)
    # compute velocity norm
    df['vel_norm'] = np.sqrt(df['vel_x'].to_numpy()**2 + df['vel_y'].to_numpy()**2)
    # compute rotation on LU
    rotated_LU = R@LU
    rotated_LU[0] -= cfg.MACROPROPS.COLS

    return df, rotated_LU

if __name__ == '__main__':
    # Example code to compute macropros of aggregated data and plot this in grid
    cfg = getYamlConfig()
    filename = "datasets/ATC/aggData/raw30minData4UT.csv"

    df = pd.read_csv(filename, index_col=0)
    df['time'] = pd.to_datetime(df['time'])
    data, rLU = preProcessData(df, LU=cfg.MACROPROPS.LU, cfg=cfg)
    plotPeopleDensity(data['pos_x'], data['pos_y'], LU=rLU, samplesToPlot=10000, title="Data after preprocess", saveImg=False)
    dataf = filterDataByLU(data, cfg, LU=rLU, showMsg=False)
    plotPeopleDensity(dataf['pos_x'], dataf['pos_y'], LU=rLU, samplesToPlot=10000, title="Data after filtering by LU", customScale=False, saveImg=False)
    dataByTime = filterDataByTime(dataf, time=T_INIT, cfg=cfg, showMsg=True)
    rho, mu_vx, mu_vy, sigma2_v = getMacroPropertiesAtTimeStamp(dataByTime, cfg, LU=rLU)
    crowd = Crowd(rho=rho, mu_v=np.stack((mu_vx, mu_vy), axis=0), sigma2_v=sigma2_v)
    rhoInFrame=np.sum(rho).astype(int)
    macropros_fig = drawMacroProps(crowd=crowd, info=[1, rhoInFrame], maxRho=2, saveImg=True)