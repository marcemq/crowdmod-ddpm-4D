import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import logging
import torch
import sys
from torchvision.utils import make_grid
from torch.cuda import amp
import torch.nn.functional as F

import numpy as np
import wandb
from matplotlib import pyplot as plt
from utils.myparser import getYamlConfig
from utils.dataset import getDataset
#from utils.train import train
from utils.crowd import plotPredictedMacroprops
from models.diffusion.forward import ForwardSampler
from models.diffusion.ddpm import DDPM

import torch.nn as nn

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("logs/CrowdPredictionDDPM.log"),
                        logging.StreamHandler(sys.stdout)]
                    )

def inverseTransform(y, stats):
    stats = stats[0,:,:]
    logging.info(f'stats shape:{stats.shape}')
    logging.info(f'y shape:{y.shape}')
    for i in range(1, len(stats)-1):
        y[:,i,:,:,:] = (y[:,i,:,:,:] + 1)/2 * (stats[i,3] - stats[i,2]) + stats[i,2]
    return y

def getGrid(x, showGrid=False):
    "x shape: BxCHxRxCxObsLen"
    # Set as grid and show
    mp_as_img = x[1,:,:,:,:].permute(3,0,1,2)
    mp_as_img = mp_as_img[:,0:3,:,:]
    print(f'first element in batch:{mp_as_img.shape}')
    grid_img = make_grid(mp_as_img, nrow=6, padding=True, pad_value=1, normalize=True)
    if showGrid:
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.axis("off")
        plt.show()
    return grid_img

        
# def train_step(batch:torch.Tensor, model:nn.Module, forwardsampler:DDPM):
#     # Sample a timestep uniformly
#     t = torch.randint(low=0, high=forwardsampler.timesteps, size=(batch.shape[0],), device=batch.device)
#     # Apply forward noising process on original images, up to step t (sample from q(x_t|x_0))
#     x_noisy, eps_true = forwardsampler(batch, t)
#     with amp.autocast():
#         # Our prediction for the denoised image
#         eps_predicted = model(x_noisy, t)
#         # Deduce the loss
#         loss          = F.mse_loss(eps_predicted, eps_true)
#     return loss


def predTraining(cfg, filenames, show_losses_plot=False):
    # set fixed random seed
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # detect if CUDA is available or not
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dtype = torch.cuda.FloatTensor # computation in GPU
    else:
        dtype = torch.float64
    # Get batched datasets ready to iterate
    batched_train_data, batched_val_data, batched_test_data = getDataset(cfg, filenames)

    # print("Number of batches: ",len(batched_train_data))
    for batch_train_idx, (x_train, y_train, _) in enumerate(batched_train_data):
         print(f'x_train batch shape:{x_train.shape}')
         print(f'y_train batch shape:{y_train.shape}')
         getGrid(x_train, showGrid=False)
         break

    # Take a batch of images
    x_train, y_train, stats = next(iter(batched_train_data))
    x_train, y_train = x_train.float(), y_train.float()
    x_train, y_train = x_train.to(device=device), y_train.to(device=device) 
    # Instantiate a forward sampler as above
    fwdSampler = ForwardSampler(1000)

    # Keeping the results for visualization
    noisy_images = []

    # Timesteps at which we will visualize the diffusion effect
    specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]

    for timestep in specific_timesteps:
        timestep = torch.as_tensor(timestep, dtype=torch.long)
        x0s      = x_train
        # Apply forward diffusion
        xts, _   = fwdSampler(x0s, timestep)
        # Rescale the images for visualization
        xts      = inverseTransform(xts, stats)
        xts      = getGrid(xts)
        noisy_images.append(xts)

    # Plot and see samples at different timesteps
    fig, ax = plt.subplots(len(noisy_images), 1, figsize=(9, 11), facecolor='white')
    fig.subplots_adjust(hspace=0.5)  # You can adjust the value as needed

    # Display the results column by column
    for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
        ax[i].imshow(noisy_sample.permute(1, 2, 0))
        ax[i].set_title(f"t={timestep}", fontsize=10)
        ax[i].axis("off")
        ax[i].grid(False)

    plt.suptitle("Forward Diffusion Process", y=0.95)
    plt.axis("off")
    plt.show()

    return 1

if __name__ == '__main__':
    cfg = getYamlConfig()
    filenames = cfg.SUNDAY_DATA_LIST
    filenames = [filename.replace(".csv", ".pkl") for filename in filenames]
    filenames = [ os.path.join(cfg.PICKLE.PICKLE_DIR, filename) for filename in filenames if filename.endswith('.pkl')]
    predTraining(cfg, filenames)
