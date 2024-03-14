import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import logging
import torch
import sys
from torchvision.utils import make_grid

from matplotlib import pyplot as plt
from utils.myparser import getYamlConfig
from utils.dataset import getDataset
from models.diffusion.forward import ForwardSampler

def inverseTransform(y, stats):
    stats = stats[0,:,:]
    for i in range(1, len(stats)-1):
        y[:,i,:,:] = (y[:,i,:,:] + 1)/2 * (stats[i,3] - stats[i,2]) + stats[i,2]
    return y

def predTraining(cfg, filenames, show_losses_plot=False):
    # set fixed random seed
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get batched datasets ready to iterate
    batched_train_data, _, _ = getDataset(cfg, filenames)
    # Take a batch of images
    x_train, y_train, stats = next(iter(batched_train_data))
    print(f'Shape of x_train data:{x_train.shape}')
    print(f'Shape of y_train data:{y_train.shape}')
    x_train, y_train = x_train.float(), y_train.float()
    x_train, y_train = x_train.to(device=device), y_train.to(device=device)

    # Instantiate a forward sampler
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
        # Apply inverse transform
        xts      = inverseTransform(xts, stats)
        noisy_images.append(xts)

    # Plot and see samples at different timesteps
    fig, ax = plt.subplots(len(noisy_images), 1, figsize=(5, 11), facecolor='white')
    fig.subplots_adjust(hspace=0.5)
    print(f'Shape of  batch of noisy_images:{noisy_images[0].shape}')
    # Display the results row by row
    for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
        one_noisy_sample = noisy_sample[0]
        one_noisy_sample_gray = torch.squeeze(one_noisy_sample[0:1,:,:], axis=0)
        ax[i].imshow(one_noisy_sample_gray, cmap='gray')
        ax[i].set_title(f"t={timestep}", fontsize=10)
        ax[i].axis("off")
        ax[i].grid(False)

    plt.suptitle("Forward Diffusion Process", y=0.95)
    plt.axis("off")
    plt.show()
    fig.savefig("images/macroPropsFwdExample.svg", format='svg', bbox_inches='tight')

if __name__ == '__main__':
    cfg = getYamlConfig()
    filenames = cfg.SUNDAY_DATA_LIST
    filenames = [filename.replace(".csv", ".pkl") for filename in filenames]
    filenames = [ os.path.join(cfg.PICKLE.PICKLE_DIR, filename) for filename in filenames if filename.endswith('.pkl')]
    predTraining(cfg, filenames)
