import argparse
import os
import pandas as pd
import numpy as np
import logging
import pickle
import sys
from utils.plot import drawMacroProps, drawPredMacroProps
from datetime import datetime
from matplotlib import pyplot as plt
import imageio
import wandb

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("logs/CrowdMacroprops.log"),
                        logging.StreamHandler(sys.stdout)]
                    )

class Crowd(object):
    def __init__(self, rows=12, cols=36, rho=None, mu_v=None, sigma2_v=None):
        self.rows = rows
        self.cols = cols
        # macroscopic properties
        self.rho = rho
        self.mu_v = mu_v
        self.sigma2_v = sigma2_v

    def setMacroProps(self, rho, mu_v, sigma2_v):
        self.rho = rho
        self.mu_v = mu_v
        self.sigma2_v = sigma2_v

def getInitIdx(pklFile, aggDataDir, t_init):
    filename = os.path.splitext(os.path.basename(pklFile))[0]
    aggData = os.path.join(aggDataDir, filename+'.csv')
    df = pd.read_csv(aggData, index_col=0)
    df['time'] = pd.to_datetime(df['time'])
    t_min = df['time'].min()
    t_diff_sec = (datetime.strptime(t_init, '%Y-%m-%d %H:%M:%S') - t_min).seconds
    initIdx = (t_diff_sec*2)//20
    logging.info("real index to start at:{}".format(initIdx))
    return initIdx

def getMaxRho(seq_per_file, initIdx=0, n_Frames=None):
    maxRho = 0
    i_frame, maxRhoFrame = 0, 0
    for i in range(initIdx, len(seq_per_file)):
        for j in range(seq_per_file.shape[-1]):
            data = seq_per_file[i,:,:,:,j]
            maxRhoTmp=np.max(data[0,:,:])
            if maxRhoTmp > maxRho:
                maxRho = maxRhoTmp
                maxRhoFrame = i_frame
            i_frame+=1
            if n_Frames is not None and i_frame == n_Frames:
                break
        if n_Frames is not None and i_frame == n_Frames:
                break
    return maxRho, maxRhoFrame

def plotMacropropsFromFile(pklFile, aggDataDir, gifName, t_init, fps=3, n_Frames=100):
    """
    A function to draw the macroproperties of a given file.
    Args:
    - pklFile (path str): macroproperties pickle file for a complete day
    - aggDir (path str): aggregation data dir
    - t_init (date): init time to start the drawing
    - n_frames (int): numbers of frames to be drawn after t_init
    """
    i_frame = 0
    initIdx = getInitIdx(pklFile, aggDataDir, t_init)
    fig_list = []
    try:
        with open(pklFile, "rb") as file:
            seq_per_file = pickle.load(file)
            logging.info("Shape of seq_per_file:{}".format(seq_per_file.shape))
            maxRho, maxRhoFrame = getMaxRho(seq_per_file, initIdx, n_Frames)
            logging.info("Max rho value for gif:{}, at frame:{}".format(maxRho, maxRhoFrame))
    except MemoryError:
        logging.info("MemoryError: Unable to load pickle data due to memory issues.")
    except Exception as e:
       logging.info(f"An error occurred while loading pkl file: {str(e)}")

    for i in range(initIdx, len(seq_per_file)):
        for j in range(seq_per_file.shape[-1]):
            #logging.info("Macropros at seq-{}, t-{}".format(i,j))
            data = seq_per_file[i,:,:,:,j]
            crowd = Crowd(rho=data[0,:,:], mu_v=data[1:3,:,:], sigma2_v=data[3,:,:])
            rhoInFrame=np.sum(data[0,:,:]).astype(int)
            macropros_fig = drawMacroProps(crowd=crowd, info=[i_frame, rhoInFrame], maxRho=maxRho)
            macropros_fig.canvas.draw()
            fig_list.append(np.array(macropros_fig.canvas.renderer.buffer_rgba()))
            plt.close()
            i_frame+=1
            if i_frame == n_Frames:
                break
        logging.info("\n")
        if i_frame == n_Frames:
                break

    imageio.mimsave('gifs/'+gifName+'.gif', fig_list, format='GIF', fps=fps)

def plotAllMacropropsFromFile(pklFile, gifName, fps=3):
    """
    This function plots all macro-properties found in pklFile and saves it
    into a gif.
    """
    pickle_in = open(pklFile,"rb")
    macropros = pickle.load(pickle_in)
    i_frame = 0
    fig_list = []
    maxRho, maxRhoFrame = getMaxRho(macropros)
    logging.info("Max rho value for gif:{}, at frame:{}".format(maxRho, maxRhoFrame))

    for i in range(len(macropros)):
        for j in range(macropros.shape[-1]):
            i_frame += 1
            data = macropros[i,:,:,:,j]
            crowd = Crowd(rho=data[0,:,:], mu_v=data[1:3,:,:], sigma2_v=data[3,:,:])
            macropros_fig = drawMacroProps(crowd=crowd, i_frame=i_frame, maxRho=maxRho)
            macropros_fig.canvas.draw()
            fig_list.append(np.array(macropros_fig.canvas.renderer.buffer_rgba()))
            plt.close()

    imageio.mimsave('gifs/'+gifName+'.gif', fig_list, format='GIF', fps=fps)

def plotPredictedMacroprops(y_hat, y_gt, gifName, drawUncGt, drawUncHat, fps=2):
    """
    This function plots predicted macro-properties for a given sequence and its GT
    """
    fig_list = []
    y_hat = np.array(y_hat.cpu())
    y_gt = np.array(y_gt.cpu())
    rho_hat = np.max(y_hat[0,:,:,:])
    rho_gt = np.max(y_gt[0,:,:,:])
    maxRho = rho_hat if rho_hat > rho_gt else rho_gt
    for frame in range(y_hat.shape[-1]):
        data_hat = y_hat[:,:,:,frame]
        data_gt = y_gt[:,:,:,frame]
        crowd_hat = Crowd(rho=data_hat[0,:,:], mu_v=data_hat[1:3,:,:], sigma2_v=data_hat[3,:,:])
        crowd_gt = Crowd(rho=data_gt[0,:,:], mu_v=data_gt[1:3,:,:], sigma2_v=data_gt[3,:,:])
        rhoInFrameHat = np.sum(data_hat[0,:,:]).astype(int)
        rhoInFrameGt = np.sum(data_gt[0,:,:]).astype(int)
        macropros_fig = drawPredMacroProps(crowd_hat=crowd_hat, crowd_gt=crowd_gt, info=[frame, rhoInFrameHat, rhoInFrameGt], maxRho=maxRho, drawUncGt=drawUncGt, drawUncHat=drawUncHat)
        macropros_fig.canvas.draw()
        fig_list.append(np.array(macropros_fig.canvas.renderer.buffer_rgba()))
        plt.close()

    imageio.mimsave('gifs/'+gifName+'.gif', fig_list, format='GIF', fps=fps)
    wandb.log({gifName: wandb.Video('gifs/'+gifName+'.gif', fps=fps, format="gif")})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to plot macro-properties for a given file.")
    parser.add_argument('--pkl-file', type=str, default='/Users/marcemq/devel/coolName/pickle/atc-20121114.pkl', help='macro-properties pickle file')
    parser.add_argument('--agg-dir', type=str, default='/Users/marcemq/Documents/PHD/outData/',help='aggregation data dir')
    parser.add_argument('--t-init', type=str, default='2012-11-14 12:00:10', help='init time from where to plot macro-pros')
    parser.add_argument('--n-frames', type=int, default=120, help='numbers of frames to be drawn after t_init')
    args = parser.parse_args()

    plotMacropropsFromFile(pklFile=args.pkl_file, aggDataDir=args.agg_dir, t_init=args.t_init, n_Frames=args.n_frames, gifName='macroprops', fps=2)
    #ploAlltMacropropsFromFile(pklFile='/Users/marcemq/devel/coolName/pickle/framesUT.pkl', gifName='macropropsFromUT', fps=1)