import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class MacropropPlotter:
    def __init__(self, cfg, output_dir, arch="DDPM-UNet", velScale=0.5, velUncScale=1.0, headwidth=5):
        self.output_dir = output_dir
        self.dataset_name = cfg.DATASET.NAME
        self.samples4plot = cfg.MODEL.NSAMPLES4PLOTS
        self.past_len   = cfg.DATASET.PAST_LEN
        self.future_len = cfg.DATASET.FUTURE_LEN
        self.sampler    = cfg.MODEL.DDPM.SAMPLER
        self.cols       = cfg.MACROPROPS.COLS
        self.rows       = cfg.MACROPROPS.ROWS
        self.arch       = arch
        self.velScale   = velScale
        self.velUncScale = velUncScale
        self.headwidth   = headwidth

    def _get_j_indexes(self, plotPast):
        """
        Get frame indexes to show depending on plot mode.
        """
        past_indexes = list(range(self.past_len))
        future_indexes = list(range(self.past_len, self.past_len + self.future_len))

        if plotPast == "Last2":
            j_indexes = past_indexes[-2:]
        elif plotPast == "Alternate":
            j_indexes = past_indexes[::2]
            if past_indexes[-1] not in j_indexes:
                j_indexes[-1] = past_indexes[-1]
        else:
            j_indexes = past_indexes

        j_indexes.extend(future_indexes)
        return j_indexes

    def _get_rho_limits(self, seq_frames, j_indexes):
        """
        Return global min/max rho values for consistent color scaling.
        """
        rho_min, rho_max = 0, float('-inf')
        for i in range(self.samples4plot * 2):
            one_seq_img = seq_frames[i]
            for j in j_indexes:
                one_sample_img = one_seq_img[:, :, :, j]
                rho = torch.squeeze(one_sample_img[0:1, :, :], axis=0)
                rho_max = max(rho_max, torch.max(rho).item())
        return rho_min, rho_max

    def plotStatic(self, seq_frames, match, plotMprop, plotPast):
        if plotMprop=="Density":
            title = f"Sampling density with {self.arch} architecture\nPast Len:{self.past_len} and Future Len:{self.future_len}"
            figName = f"{self.output_dir}/mpSampling_{self.arch}_4Density_{match.group()}.svg"
        elif plotMprop=="Uncertainty":
            title = f"Sampling uncertainty with {self.arch} architecture\nPast Len:{self.past_len} and Future Len:{self.future_len}"
            figName = f"{self.output_dir}/mpSampling_{self.arch}_4Uncertainty_{match.group()}.svg"
        else:
            title =  f"Sampling macroprops with {self.arch} architecture\nPast Len:{self.past_len} and Future Len:{self.future_len}"
            figName= f"{self.output_dir}/mpSampling_{self.arch}_{match.group()}.svg"

        j_indexes = self._get_j_indexes(plotPast)
        rho_min, rho_max = self._get_rho_limits(seq_frames, j_indexes)

        static_samples4plot = 4
        fig, ax = plt.subplots(static_samples4plot*2, len(j_indexes), figsize=(10,8), facecolor='white')
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        for i in range(static_samples4plot*2):
            one_seq_img = seq_frames[i]
            for ind, j in enumerate(j_indexes):
                if ind == 0:
                    label = f"GT\nseq-{i // 2 + 1}" if (i + 1) % 2 == 0 else f"Pred\nseq-{i // 2 + 1}"
                    fig.text(0.11, 0.845 - i / (static_samples4plot * 2 + 4.6), label, fontsize=8, ha='center', va='center', rotation=90)

                one_sample_img = one_seq_img[:,:,:,j].cpu()
                rho = torch.squeeze(one_sample_img[0:1,:,:], axis=0)
                mu_v = torch.squeeze(one_sample_img[1:3,:,:], axis=0)
                sigma2_v = torch.squeeze(one_sample_img[3:4,:,:], axis=0)

                # Plot density
                axp = ax[i, ind].matshow(rho, cmap=plt.cm.Blues, vmin=rho_min, vmax=rho_max)
                # Plot density and velocity vectors
                if plotMprop=="Density&Vel":
                    Q = ax[i, ind].quiver(mu_v[0,:,:], -mu_v[1,:,:], color='green', angles='xy',scale_units='xy', scale=self.velScale, minshaft=3.5, width=0.009, headwidth=self.headwidth)
                # Plot density and velocity uncertainty
                if plotMprop=="Uncertainty":
                    x, y = np.mgrid[0:self.cols, 0:self.rows]
                    for ii in range(self.rows):
                        for jj in range(self.cols):
                            center = (x[jj,ii], y[jj,ii])
                            circle = plt.Circle(center, self.velUncScale*np.sqrt(sigma2_v[ii,jj]), fill=False, color='green', lw=0.7)
                            axp.axes.add_artist(circle)

                ax[i, ind].axis('off')
                ax[i, ind].grid(False)

        # Color bar for density rho
        cbar = fig.colorbar(axp, ax=ax.ravel().tolist(), pad=0.04, shrink=0.45, orientation="horizontal")
        cbar.set_label('Density rho', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        plt.suptitle(title, y=0.95)
        plt.axis("off")
        fig.savefig(figName, format='svg', bbox_inches='tight')

    def plotDynamic(self, seq_frames):
        j_indexes = self._get_j_indexes(plotPast="All")
        rho_min, rho_max = self._get_rho_limits(seq_frames, j_indexes)
        title =  f"Sampling macroprops with {self.arch} architecture\nPast Len:{self.past_len} and Future Len:{self.future_len}"

        # Iterate over each sequence to create a GIF for each
        for i in range(self.samples4plot*2):
            if self.dataset_name in ["ATC", "ATC4TEST", "HERMES-BO"]:
                fig, ax = plt.subplots(1, 1, figsize=(7, 4), facecolor='white')
            elif self.dataset_name in ["HERMES-CR-120", "HERMES-CR-120-OBST"]:
                fig, ax = plt.subplots(1, 1, figsize=(4, 5), facecolor='white')
            else:
                logging.error("Dynamic plot: Dataset not supported")
            fig.subplots_adjust(hspace=0.1, wspace=0.1)

            # Set up the initial plot and color bar
            one_seq_img = seq_frames[i]
            j = j_indexes[0]
            one_sample_img = one_seq_img[:, :, :, j].cpu()
            rho = torch.squeeze(one_sample_img[0:1, :, :], axis=0)
            mu_v = torch.squeeze(one_sample_img[1:3, :, :], axis=0)

            # Initial plot and color bar
            axp = ax.matshow(rho, cmap=plt.cm.Blues, vmin=rho_min, vmax=rho_max)
            Q = ax.quiver(mu_v[0, :, :], -mu_v[1, :, :], color='green', angles='xy', scale_units='xy', scale=self.velScale, minshaft=3.5, width=0.009, headwidth=self.headwidth)
            # color bar setup
            cbar = fig.colorbar(axp, ax=ax, orientation='vertical', fraction=0.015)
            cbar.set_label('Density rho', fontsize=11)
            cbar.ax.tick_params(labelsize=10)

            plt.title(title, fontsize=12)
            frame_text = ax.text(0.5, -0.15, '', transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')

            def update(frame):
                j = j_indexes[frame]
                one_sample_img = one_seq_img[:, :, :, j].cpu()
                rho = torch.squeeze(one_sample_img[0:1, :, :], axis=0)
                mu_v = torch.squeeze(one_sample_img[1:3, :, :], axis=0)
                # Update the plot without clearing the axis
                axp.set_array(rho)
                Q.set_UVC(mu_v[0, :, :], -mu_v[1, :, :])
                # Update the frame number text and color accordingly
                if (i + 1) % 2 == 0:
                    frame_text.set_color('black')
                else:
                    if frame < self.past_len:
                        frame_text.set_color('black')
                    else:
                        frame_text.set_color('blue')
                frame_text.set_text(f'Frame: {frame + 1}/{len(j_indexes)}')

            # Set up animation for the current sequence
            ani = animation.FuncAnimation(fig, update, frames=len(j_indexes), repeat=True)
            # Save each sequence as a separate GIF
            gif_name = f"{self.output_dir}/mprops_GT_seq_{i // 2 + 1}.gif" if (i + 1) % 2 == 0 else f"{self.output_dir}/mprops_seq_{i // 2 + 1}.gif"
            ani.save(gif_name, writer=PillowWriter(fps=2))
            plt.close(fig)

    def plotDensityOverTime(self, seq_frames):
        logging.info(f'Seq frame shape: {seq_frames[0].shape}')
        _, _, _, L = seq_frames[0].shape  # Get sequence length dynamically

        for i in range(self.samples4plot):
            rho_pred = seq_frames[2 * i][0, :, :, :].sum(dim=(0, 1)).cpu().numpy()
            rho_gt = seq_frames[2 * i + 1][0, :, :, :].sum(dim=(0, 1)).cpu().numpy()

            # Create time steps
            frames = np.arange(1, L + 1)

            # Plot both in the same figure
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(frames[0:self.past_len], rho_gt[0:self.past_len], color="blue", marker="o", label="Past")
            ax.scatter(frames[self.past_len:], rho_pred[self.past_len:], color="red", marker="o", label="Predicted")
            ax.scatter(frames[self.past_len:], rho_gt[self.past_len:], color="green", marker="o", label="Ground Truth")

            ax.set_xlabel("Frame")
            ax.set_ylabel("Sum of density ρ")
            ax.set_title("Sum of density over time")
            ax.legend()

            plot_name = f"{self.output_dir}/rho_seq_{i + 1}.png"
            fig.savefig(plot_name)
            plt.close(fig)  # Avoid excessive memory usage

        logging.info(f"Density plots saved in {self.output_dir}")