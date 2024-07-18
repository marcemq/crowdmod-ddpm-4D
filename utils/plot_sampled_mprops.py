import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.crowd import Crowd
from utils.plot import drawMacroProps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def plotDensity(seq_images, cfg, match):
    # Plot and see samples at different timesteps
    fig, ax = plt.subplots(cfg.DIFFUSION.NSAMPLES*2, cfg.DATASET.PAST_LEN+cfg.DATASET.FUTURE_LEN, figsize=(13,7), facecolor='white')
    fig.subplots_adjust(hspace=0.3)
    for i in range(cfg.DIFFUSION.NSAMPLES*2):
        one_seq_img = seq_images[i]
        for j in range(cfg.DATASET.PAST_LEN+cfg.DATASET.FUTURE_LEN):
            if j==0:
                if (i+1)%2==0:
                    ax[i,j].set_title(f" GT sequence-{i//2+1}", fontsize=9)
                else:
                    ax[i,j].set_title(f"Pred sequence-{i//2+1}", fontsize=9)
            one_sample_img = one_seq_img[:,:,:,j]
            one_sample_img_gray = torch.squeeze(one_sample_img[0:1,:,:], axis=0)
            ax[i,j].imshow(one_sample_img_gray.cpu(), cmap='gray')
            ax[i,j].axis("off")
            ax[i,j].grid(False)

    plt.suptitle(f"Sampling for diffusion process using {cfg.DIFFUSION.SAMPLER}\nPast Len:{cfg.DATASET.PAST_LEN} and Future Len:{cfg.DATASET.FUTURE_LEN}", y=0.95)
    plt.axis("off")
    plt.show()
    fig.savefig(f"images/mpSampling_4Density_{cfg.DIFFUSION.SAMPLER}_{match.group()}.svg", format='svg', bbox_inches='tight')

def _get_j_indexes(cfg, plotPast):
    past_indexes = list(range(cfg.DATASET.PAST_LEN))
    future_indexes = list(range(cfg.DATASET.PAST_LEN, cfg.DATASET.PAST_LEN + cfg.DATASET.FUTURE_LEN))

    if plotPast == "Last2":
        j_indexes = past_indexes[-2:]
    if plotPast == "Alternate":
        j_indexes = past_indexes[::2]
        if past_indexes[-1] not in j_indexes:
            j_indexes[-1] = past_indexes[-1]

    j_indexes.extend(future_indexes)
    return j_indexes

def _get_rho_limits(cfg, seq_images, j_indexes):
    rho_min, rho_max = 0, float('-inf')  # Set the color limits here

    for i in range(cfg.DIFFUSION.NSAMPLES * 2):
        one_seq_img = seq_images[i]
        for j in j_indexes:
            one_sample_img = one_seq_img[:, :, :, j]
            rho = torch.squeeze(one_sample_img[0:1, :, :], axis=0)
            rho_max = max(rho_max, torch.max(rho).item())

    return rho_min, rho_max

def plotAllMacroprops(seq_images, cfg, match, plotPast):
    j_indexes = _get_j_indexes(cfg, plotPast)
    rho_min, rho_max = _get_rho_limits(cfg, seq_images, j_indexes)

    fig, ax = plt.subplots(cfg.DIFFUSION.NSAMPLES*2, len(j_indexes), figsize=(10,8), facecolor='white')
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i in range(cfg.DIFFUSION.NSAMPLES*2):
        one_seq_img = seq_images[i]
        for ind, j in enumerate(j_indexes):
            if ind == 0:
                label = f"GT\nseq-{i // 2 + 1}" if (i + 1) % 2 == 0 else f"Pred\nseq-{i // 2 + 1}"
                fig.text(0.11, 0.845 - i / (cfg.DIFFUSION.NSAMPLES * 2 + 4.6), label, fontsize=8, ha='center', va='center', rotation=90)

            one_sample_img = one_seq_img[:,:,:,j]
            rho = torch.squeeze(one_sample_img[0:1,:,:], axis=0)
            mu_v = torch.squeeze(one_sample_img[1:3,:,:], axis=0)
            sigma2_v = torch.squeeze(one_sample_img[3:4,:,:], axis=0)

            axp = ax[i, ind].matshow(rho, cmap=plt.cm.Blues, vmin=rho_min, vmax=rho_max)
            Q = ax[i, ind].quiver(mu_v[0,:,:], -mu_v[1,:,:], color='green', angles='xy',scale_units='xy', scale=1)

            x, y = np.mgrid[0:cfg.MACROPROPS.COLS, 0:cfg.MACROPROPS.ROWS]
            for ii in range(cfg.MACROPROPS.ROWS):
                for jj in range(cfg.MACROPROPS.COLS):
                    center = (x[jj,ii]+mu_v[0,ii,jj], y[jj,ii]-mu_v[1,ii,jj])
                    circle = plt.Circle(center, 2*np.sqrt(sigma2_v[ii,jj]), fill=False, color='green')
                    Q.axes.add_artist(circle)

            ax[i, ind].axis('off')
            ax[i, ind].grid(False)

    # Color bar for density rho
    cbar = fig.colorbar(axp, ax=ax.ravel().tolist(), pad=0.04, shrink=0.45, orientation="horizontal")
    cbar.set_label('Density rho', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.suptitle(f"Sampling for diffusion process using {cfg.DIFFUSION.SAMPLER}\nPast Len:{cfg.DATASET.PAST_LEN} and Future Len:{cfg.DATASET.FUTURE_LEN}", y=0.95)
    plt.axis("off")
    fig.savefig(f"images/mpSampling_{cfg.DIFFUSION.SAMPLER}_{match.group()}.svg", format='svg', bbox_inches='tight')