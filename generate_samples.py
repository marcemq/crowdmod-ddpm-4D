import torch
import matplotlib.pyplot as plt
import numpy as np
import os, re
from models.generate import generate_ddpm, generate_ddim

from models.unet import MacroprosDenoiser
from models.diffusion.ddpm import DDPM
from utils.dataset import getDataset
from utils.myparser import getYamlConfig
from torchvision.utils import make_grid

def inverseTransform(y, stats):
    stats = stats[0,:,:]
    print(f'stats shape:{stats.shape}')
    print(f'y shape:{y.shape}')
    for i in range(1, len(stats)-1):
        y[:,i,:,:,:] = (y[:,i,:,:,:] + 1)/2 * (stats[i,3] - stats[i,2]) + stats[i,2]
    return y

def getGrid(x, cols, mode="RGB", showGrid=False):
    "x shape: BxCHxRxCxObsLen"
    # Set as grid and show
    mp_as_img = x[1,:,:,:,:].permute(3,0,1,2)
    if mode == "RGB":
        mp_as_img = mp_as_img[:,0:3,:,:]
    elif mode == "GRAY":
        mp_as_img = mp_as_img[:,0:1,:,:]
    print(f'first element in batch:{mp_as_img.shape}')
    grid_img = make_grid(mp_as_img, nrow=cols, padding=True, pad_value=1, normalize=True)
    if showGrid:
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.axis("off")
        plt.show()
    return grid_img

def generate_samples(cfg, filenames):
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get batched datasets ready to iterate
    batched_train_data, _, _ = getDataset(cfg, filenames)

    # Instanciate the UNet for the reverse diffusion
    denoiser = MacroprosDenoiser(num_res_blocks = cfg.MODEL.NUM_RES_BLOCKS,
                                base_channels           = cfg.MODEL.BASE_CH,
                                base_channels_multiples = cfg.MODEL.BASE_CH_MULT,
                                apply_attention         = cfg.MODEL.APPLY_ATTENTION,
                                dropout_rate            = cfg.MODEL.DROPOUT_RATE,
                                time_multiple           = cfg.MODEL.TIME_EMB_MULT)
    model_fullname = cfg.MODEL.SAVE_DIR+(cfg.MODEL.MODEL_NAME.format(cfg.TRAIN.EPOCHS))
    model_fullname = "saved_models/UNet_Macropros_E1000_LR0000005.pth"
    print(f'model full name:{model_fullname}')
    denoiser.load_state_dict(torch.load(model_fullname, map_location=torch.device('cpu'))['model'])
    denoiser.to(device)

    # Instantiate the diffusion model 
    timesteps=cfg.DIFFUSION.TIMESTEPS
    diffusionmodel = DDPM(timesteps=cfg.DIFFUSION.TIMESTEPS)
    diffusionmodel.to(device)
    noisy_images = []
    taus = 1
    for batch in batched_train_data:
        x_train, y_train, stats = batch
        if cfg.DIFFUSION.SAMPLER == "DDPM":
            x = generate_ddpm(denoiser, diffusionmodel, cfg, device).cpu() # AR review .cpu() call here
        elif cfg.DIFFUSION.SAMPLER == "DDIM":
            taus = np.arange(0,timesteps,cfg.DIFFUSION.DDIM_DIVIDER)
            print(f'taus:{taus}')
            x, xnoisy_over_time = generate_ddim(denoiser,taus,diffusionmodel,cfg,device) # AR review .cpu() call here
        else:
            print(f"{cfg.DIFFUSION.SAMPLER} sampler not supported")

        for i in range(len(xnoisy_over_time)):
            xts      = getGrid(xnoisy_over_time[i], cfg.DATASET.OBS_LEN, mode="GRAY")
            noisy_images.append(xts)
        
        # Plot and see samples at different timesteps
        fig, ax = plt.subplots(len(noisy_images), 1, figsize=(12, 11), facecolor='white')
        fig.subplots_adjust(hspace=0.5)

        # Display the results row by row
        for i, (timestep, noisy_sample) in enumerate(zip(reversed(taus), noisy_images)):
            ax[i].imshow(noisy_sample.permute(1, 2, 0))
            ax[i].set_title(f"t={timestep}", fontsize=10)
            ax[i].axis("off")
            ax[i].grid(False)

        plt.suptitle("Sampling for diffusion process", y=0.95)
        plt.axis("off")
        plt.show()
        match = re.search(r'_E\d+_LR\d+', model_fullname)
        fig.savefig(f"images/mpSampling_{match.group()}.svg", format='svg', bbox_inches='tight')
        break

if __name__ == '__main__':
    cfg = getYamlConfig()
    filenames = cfg.SUNDAY_DATA_LIST
    filenames = [filename.replace(".csv", ".pkl") for filename in filenames]
    filenames = [ os.path.join(cfg.PICKLE.PICKLE_DIR, filename) for filename in filenames if filename.endswith('.pkl')]
    generate_samples(cfg, filenames)

