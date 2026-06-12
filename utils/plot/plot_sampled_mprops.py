import logging, re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage.metrics import structural_similarity as ssim

FIGSIZE_MAP = {
    "ATC":                  (7, 4),
    "ATC4TEST":             (7, 4),
    "HERMES-BO":            (7, 4),
    "HERMES-BN":            (4, 7),
    "HERMES-CR-90":         (5, 4),
    "HERMES-CR-90-OBST":    (5, 4),
}

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
        self.params     = cfg.METRICS
        self.eps        = cfg.MACROPROPS.EPS
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

    def plotDynamic(self, seq_frames, seq_psnr, seq_ssim):
        j_indexes = self._get_j_indexes(plotPast="All")
        rho_min, rho_max = self._get_rho_limits(seq_frames, j_indexes)
        title =  f"Sampling macroprops with {self.arch} architecture\nPast Len:{self.past_len} and Future Len:{self.future_len}"
        # Iterate over each sequence to create a GIF for each
        for i in range(self.samples4plot*2):
            figsize = FIGSIZE_MAP.get(self.dataset_name)
            if figsize is None:
                logging.info("Dataset not supported!!!!")
                continue
            fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
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
                    psnr_text = ""
                    ssim_text = ""
                else:
                    seq_idx = i // 2
                    psnr_text = (f'psnr_rho:{seq_psnr[seq_idx, frame, 0]:.3f}, '
                                 f'psnr_vx:{seq_psnr[seq_idx, frame, 1]:.3f}, '
                                 f'psnr_vy:{seq_psnr[seq_idx, frame, 2]:.3f}'
                                )
                    ssim_text = (f'ssim_rho:{seq_ssim[seq_idx, frame, 0]:.3f}, '
                                 f'ssim_vx:{seq_ssim[seq_idx, frame, 1]:.3f}, '
                                 f'ssim_vy:{seq_ssim[seq_idx, frame, 2]:.3f}'
                                )
                    if frame < self.past_len:
                        frame_text.set_color('black')
                    else:
                        frame_text.set_color('blue')
                frame_text.set_text(f'Frame: {frame + 1}/{len(j_indexes)} \n {psnr_text} \n {ssim_text}')

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

def setup_predictions_plot(predictions, random_past_idx, random_past_samples, random_future_samples, model_fullname, plotType, plotMprop, plotPast, macropropPlotter):
    seq_frames = []
    pred_seq_list = []
    gt_seq_list   = []

    for i in range(len(random_past_idx)):
        future_sample_pred = predictions[i]
        future_sample_gt = random_future_samples[i]
        past_sample = random_past_samples[i]

        seq_pred = torch.cat([past_sample, future_sample_pred], dim=3)
        seq_gt = torch.cat([past_sample, future_sample_gt], dim=3)
        seq_frames.append(seq_pred)
        seq_frames.append(seq_gt)
        pred_seq_list.append(seq_pred)
        gt_seq_list.append(seq_gt)

    match = re.search(r'TE\d+_PL\d+_FL\d+_CE\d+_VN[FT]', model_fullname)
    seq_psnr = get_psnr_per_seq(macropropPlotter.params, pred_seq_list, gt_seq_list, macropropPlotter.eps)
    seq_ssim = get_ssim_per_seq(macropropPlotter.params, pred_seq_list, gt_seq_list)

    if plotType == "Static":
        macropropPlotter.plotStatic(seq_frames, match, plotMprop, plotPast)
    elif plotType == "Dynamic":
        macropropPlotter.plotDynamic(seq_frames, seq_psnr, seq_ssim)

    macropropPlotter.plotDensityOverTime(seq_frames)

def get_ssim_per_seq(params, pred_seq_list, gt_seq_list):
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    nsamples_ssim = np.zeros((nsamples, pred_len, params.MPROPS_COUNT))
    rho_range, vx_range, vy_range = _get_mprops_ranges(params.MPROPS_COUNT, gt_seq_list)

    for i in range(nsamples):
        one_pred_seq = pred_seq_list[i].cpu().numpy()
        one_gt_seq = gt_seq_list[i].cpu().numpy()

        for j in range(pred_len):
            frame_ssim_rho = ssim(one_gt_seq[0, :, :, j], one_pred_seq[0, :, :, j], data_range=rho_range)
            frame_ssim_vx  = ssim(one_gt_seq[1, :, :, j], one_pred_seq[1, :, :, j], data_range=vx_range)
            frame_ssim_vy  = ssim(one_gt_seq[2, :, :, j], one_pred_seq[2, :, :, j], data_range=vy_range)

            nsamples_ssim[i, j] = (frame_ssim_rho, frame_ssim_vx, frame_ssim_vy)

    return nsamples_ssim

def get_psnr_per_seq(params, pred_seq_list, gt_seq_list, eps):
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    nsamples_psnr = np.zeros((nsamples, pred_len, params.MPROPS_COUNT))

    rho_range, vx_range, vy_range = _get_mprops_ranges(params.MPROPS_COUNT, gt_seq_list)
    logging.info(f'Range of macroprops at sampling \n rho:{rho_range:.4f}, vx:{vx_range:.4f} and vy:{vy_range:.4f}')

    for i in range(nsamples):
        one_pred_seq = pred_seq_list[i].cpu().numpy()
        one_gt_seq = gt_seq_list[i].cpu().numpy()

        for j in range(pred_len):
            gt_frame   = one_gt_seq[:, :, :, j]    # (3, ROWS, COLS)
            pred_frame = one_pred_seq[:, :, :, j]  # (3, ROWS, COLS)
            mask = gt_frame[0] > 0.00001          # rho mask, shape (ROWS, COLS)

            psnr_frame_rho = _my_psnr_masked(gt_frame[0], pred_frame[0], rho_range, eps, mask)
            psnr_frame_vx  = _my_psnr_masked(gt_frame[1], pred_frame[1], vx_range,  eps, mask)
            psnr_frame_vy  = _my_psnr_masked(gt_frame[2], pred_frame[2], vy_range,  eps, mask)

            nsamples_psnr[i, j] = (psnr_frame_rho, psnr_frame_vx, psnr_frame_vy)

    return nsamples_psnr

def _get_mprops_ranges(mprops_count, gt_seq_list):
        nsamples = len(gt_seq_list)
        # Initialize arrays to store max and min values for each sample and each property
        max_vals = np.zeros((nsamples, mprops_count))
        min_vals = np.zeros((nsamples, mprops_count))

        for i, one_gt_seq in enumerate(gt_seq_list):
            # Convert the tensor to a numpy array and scale it
            one_gt_seq = one_gt_seq.cpu().numpy()

            # Calculate max and min values for rho, vx, and vy, storing them in columns
            max_vals[i, 0], min_vals[i, 0] = one_gt_seq[0].max(), one_gt_seq[0].min()  # rho
            max_vals[i, 1], min_vals[i, 1] = one_gt_seq[1].max(), one_gt_seq[1].min()  # vx
            max_vals[i, 2], min_vals[i, 2] = one_gt_seq[2].max(), one_gt_seq[2].min()  # vy

        # Compute the overall max and min values for each macro-property across all samples
        global_max_rho, global_max_vx, global_max_vy= max_vals.max(axis=0)
        global_min_rho, global_min_vx, global_min_vy= min_vals.min(axis=0)

        # Compute the range for each macro-property
        rho_range = float(global_max_rho - global_min_rho)
        vx_range  = float(global_max_vx - global_min_vx)
        vy_range  = float(global_max_vy - global_min_vy)

        return rho_range, vx_range, vy_range

def _my_psnr(y_gt, y_hat, data_range, eps):
        # Compute mean squared error
        err = np.mean((y_gt - y_hat) ** 2, dtype=np.float64)
        # Prevent overflow and division by zero
        err = max(err, eps)
        # Calculate PSNR
        tmp_num = 20 * np.log10(data_range)
        tmp_den = 10 * np.log10(err)
        psnr = tmp_num - tmp_den
        return psnr

def _my_psnr_masked(y_gt, y_hat, data_range, eps, mask):
    err = np.mean((y_gt[mask] - y_hat[mask]) ** 2, dtype=np.float64)
    err = max(err, eps)
    tmp_num = 20 * np.log10(data_range)
    tmp_den = 10 * np.log10(err)
    return tmp_num - tmp_den