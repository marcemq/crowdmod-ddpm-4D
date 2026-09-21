import logging, re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage.metrics import structural_similarity as ssim

FIGSIZE_MAP = {
    # Width includes the colorbar; height includes title + bottom text.
    "ATC":               (6.2, 2.3),
    "ATC4TEST":          (6.2, 2.3),
    "HERMES-T":          (6.2, 3.0),
    "HERMES-BO":         (6.0, 3.0),
    "HERMES-BN":         (4.2, 6.2),
    "HERMES-CR-90":      (5.8, 3.0),
    "HERMES-CR-90-OBST": (5.8, 3.0),
}

FRAME_TEXT_MAP = {
    "ATC":                  (6, 4),
    "ATC4TEST":             (6, 4),
    "HERMES-T":             (5, 4),
    "HERMES-BO":            (7, 4),
    "HERMES-BN":            (4, 7),
    "HERMES-CR-90":         (5, 4),
    "HERMES-CR-90-OBST":    (5, 4),
}

class MacropropPlotter:
    # ---- Layout constants (inches) ----
    CELL_SIZE   = 0.15   # inches per grid cell — single knob for overall resolution/density
    MIN_AXES_W, MIN_AXES_H = 2.2, 1.5
    MAX_AXES_W, MAX_AXES_H = 6.5, 4.5

    LEFT_MARGIN    = 0.55   # row tick labels
    CBAR_GAP       = 0.10
    CBAR_WIDTH     = 0.10
    CBAR_LABEL_PAD = 0.42   # "Density rho" label + colorbar ticks

    TITLE_FONTSIZE    = 12
    TITLE_TOP_PAD     = 0.06
    TITLE_LINE_H      = 0.24
    TITLE_BOTTOM_GAP  = 0.06
    TOP_TICKS_H       = 0.20

    FOOTER_TOP_GAP        = 0.05
    FOOTER_BOTTOM_PAD     = 0.04
    FOOTER_FONTSIZE_SINGLE, FOOTER_FONTSIZE_MULTI = 11, 8
    FOOTER_LINE_H_SINGLE,   FOOTER_LINE_H_MULTI   = 0.22, 0.15

    FONT_CHAR_WIDTH_EM = 0.5

    def __init__(self, cfg, output_dir, arch="DDPM-UNet", velScale=0.5, velUncScale=1.0, headwidth=5):
        self.output_dir = output_dir
        self.dataset_name = cfg.DATASET.NAME
        self.max_rho4plot = cfg.DATASET.MAX_RHO_4_PLOT
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

    def _estimate_text_width_in(self, text, fontsize):
        return len(text) * fontsize / 72.0 * self.FONT_CHAR_WIDTH_EM

    def _fit_axes_size_in(self, rows, cols):
        """Axes size in inches, exactly matching the grid's aspect ratio
        (so ax.set_aspect('equal') has no leftover slack inside its box),
        clamped to a sane visual range."""
        axes_w, axes_h = cols * self.CELL_SIZE, rows * self.CELL_SIZE
        shrink = min(self.MAX_AXES_W / axes_w, self.MAX_AXES_H / axes_h, 1.0)
        axes_w, axes_h = axes_w * shrink, axes_h * shrink
        grow = max(self.MIN_AXES_W / axes_w, self.MIN_AXES_H / axes_h, 1.0)
        return axes_w * grow, axes_h * grow

    def plotStatic(self, seq_frames, match, plotMprop, plotPast):
        if plotMprop=="Density":
            title = f"Sampling density with {self.arch}, P/F : {self.past_len}/{self.future_len}"
            figName = f"{self.output_dir}/mpSampling_{self.arch}_4Density_{match.group()}.svg"
        elif plotMprop=="Uncertainty":
            title = f"Sampling uncertainty with {self.arch}, P/F : {self.past_len}/{self.future_len}"
            figName = f"{self.output_dir}/mpSampling_{self.arch}_4Uncertainty_{match.group()}.svg"
        else:
            title =  f"Sampling macroprops with {self.arch}, P/F : {self.past_len}/{self.future_len}"
            figName= f"{self.output_dir}/mpSampling_{self.arch}_{match.group()}.svg"

        j_indexes = self._get_j_indexes(plotPast)
        rho_min, rho_max = 0, self.max_rho4plot

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

    def plotDynamic(self, seq_frames, seq_psnr, seq_masked_psnr, seq_ssim, seq_tv, show_metrics_bottom):
        j_indexes = self._get_j_indexes(plotPast="All")
        rho_min, rho_max = 0, self.max_rho4plot

        # ---- Layout computed once — identical for every GIF in this batch ----
        axes_w, axes_h = self._fit_axes_size_in(self.rows, self.cols)
        fig_w = self.LEFT_MARGIN + axes_w + self.CBAR_GAP + self.CBAR_WIDTH + self.CBAR_LABEL_PAD

        title_full = f"Sampling macroprops with {self.arch}, P/F : {self.past_len}/{self.future_len}"
        if self._estimate_text_width_in(title_full, self.TITLE_FONTSIZE) <= fig_w - 0.15:
            title_lines = [title_full]
        else:
            title_lines = [f"Sampling macroprops with {self.arch}",
                            f"P/F : {self.past_len}/{self.future_len}"]

        title_block_h = self.TITLE_TOP_PAD + len(title_lines) * self.TITLE_LINE_H

        footer_fontsize = self.FOOTER_FONTSIZE_MULTI if show_metrics_bottom else self.FOOTER_FONTSIZE_SINGLE
        footer_line_h   = self.FOOTER_LINE_H_MULTI if show_metrics_bottom else self.FOOTER_LINE_H_SINGLE
        n_footer_lines  = 5 if show_metrics_bottom else 1
        footer_block_h  = n_footer_lines * footer_line_h + self.FOOTER_BOTTOM_PAD

        # title -> [gap + top tick-label row] -> axes -> footer
        fig_h = (title_block_h + self.TITLE_BOTTOM_GAP + self.TOP_TICKS_H + axes_h + self.FOOTER_TOP_GAP + footer_block_h)

        axes_rect = [self.LEFT_MARGIN / fig_w,
                     (footer_block_h + self.FOOTER_TOP_GAP) / fig_h,
                     axes_w / fig_w, axes_h / fig_h]
        cax_rect  = [(self.LEFT_MARGIN + axes_w + self.CBAR_GAP) / fig_w,
                     axes_rect[1], self.CBAR_WIDTH / fig_w, axes_rect[3]]
        title_y   = 1.0 - self.TITLE_TOP_PAD / fig_h
        footer_y  = footer_block_h / fig_h
        title_text = "\n".join(title_lines)

        for i in range(self.samples4plot * 2):
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=120, facecolor="white")

            one_seq_img = seq_frames[i]
            j = j_indexes[0]
            one_sample_img = one_seq_img[:, :, :, j].cpu()
            rho = torch.squeeze(one_sample_img[0:1, :, :], axis=0)
            mu_v = torch.squeeze(one_sample_img[1:3, :, :], axis=0)

            ax = fig.add_axes(axes_rect)
            ax.set_aspect("equal", adjustable="box")
            cax = fig.add_axes(cax_rect)

            axp = ax.matshow(rho, cmap=plt.cm.Blues, vmin=rho_min, vmax=rho_max)
            Q = ax.quiver(mu_v[0], -mu_v[1], color="green", angles="xy", scale_units="xy",
                           scale=self.velScale, minshaft=3.5, width=0.009, headwidth=self.headwidth)
            cbar = fig.colorbar(axp, cax=cax, orientation="vertical")
            cbar.set_label("Density rho", fontsize=11)
            cbar.ax.tick_params(labelsize=10)

            fig.text(0.5, title_y, title_text, ha="center", va="top", fontsize=self.TITLE_FONTSIZE)
            frame_text = fig.text(0.5, footer_y, "", ha="center", va="top",
                                   fontsize=footer_fontsize,
                                   fontweight=None if show_metrics_bottom else "bold")

            def update(frame):
                j = j_indexes[frame]
                one_sample_img = one_seq_img[:, :, :, j].cpu()
                rho = torch.squeeze(one_sample_img[0:1, :, :], axis=0)
                mu_v = torch.squeeze(one_sample_img[1:3, :, :], axis=0)
                axp.set_array(rho)
                Q.set_UVC(mu_v[0, :, :], -mu_v[1, :, :])
                if (i + 1) % 2 == 0:
                    frame_text.set_color('black')
                    mask_psnr_text = psnr_text = ssim_text = tv_text = ""
                else:
                    seq_idx = i // 2
                    psnr_text = (f'psnr_rho:{seq_psnr[seq_idx, frame, 0]:.3f}, '
                                 f'psnr_vx:{seq_psnr[seq_idx, frame, 1]:.3f}, '
                                 f'psnr_vy:{seq_psnr[seq_idx, frame, 2]:.3f}')
                    mask_psnr_text = (f'mpsnr_rho:{seq_masked_psnr[seq_idx, frame, 0]:.3f}, '
                                 f'mpsnr_vx:{seq_masked_psnr[seq_idx, frame, 1]:.3f}, '
                                 f'mpsnr_vy:{seq_masked_psnr[seq_idx, frame, 2]:.3f}')
                    ssim_text = (f'ssim_rho:{seq_ssim[seq_idx, frame, 0]:.3f}, '
                                 f'ssim_vx:{seq_ssim[seq_idx, frame, 1]:.3f}, '
                                 f'ssim_vy:{seq_ssim[seq_idx, frame, 2]:.3f}')
                    tv_text   = (f'tv_rho:{seq_tv[seq_idx, frame, 0]:.3f}, '
                                 f'tv_vx:{seq_tv[seq_idx, frame, 1]:.3f}, '
                                 f'tv_vy:{seq_tv[seq_idx, frame, 2]:.3f}')
                    frame_text.set_color('black' if frame < self.past_len else 'blue')
                if show_metrics_bottom:
                    frame_text.set_text(f'Frame: {frame + 1}/{len(j_indexes)} \n {psnr_text} \n {mask_psnr_text} \n {ssim_text} \n {tv_text}')
                else:
                    frame_text.set_text(f'Frame: {frame + 1}/{len(j_indexes)}')

            ani = animation.FuncAnimation(fig, update, frames=len(j_indexes), repeat=True, blit=False)
            gif_name = f"{self.output_dir}/mprops_GT_seq_{i // 2 + 1}.gif" if (i + 1) % 2 == 0 else f"{self.output_dir}/mprops_seq_{i // 2 + 1}.gif"
            ani.save(gif_name, writer=PillowWriter(fps=2), dpi=120, savefig_kwargs={"facecolor": "white"})
            plt.close(fig)

    def plotDynamic_ori(self, seq_frames, seq_psnr, seq_masked_psnr, seq_ssim, seq_tv, show_metrics_bottom):
        j_indexes = self._get_j_indexes(plotPast="All")
        rho_min, rho_max = 0, self.max_rho4plot
        title =  f"Sampling macroprops with {self.arch}, P/F : {self.past_len}/{self.future_len}"
        # Iterate over each sequence to create a GIF for each
        for i in range(self.samples4plot*2):
            figsize = FIGSIZE_MAP.get(self.dataset_name)
            if figsize is None:
                logging.info("Dataset not supported!!!!")
                continue

            # The metric version needs more vertical room than the frame-only version.
            if show_metrics_bottom:
                # Keep the base FIGSIZE_MAP dimensions for the multi-line metrics block.
                fig_w, fig_h = figsize
                axes_top = 0.82
                min_axes_bottom = 0.30
                footer_gap = 0.035
                frame_fontsize = 8
            else:
                # ATC's wide 36x12 grid is compact in a 6.2 x 2.4 inch canvas.
                fig_w, fig_h = figsize
                axes_top = 0.78
                min_axes_bottom = 0.15
                footer_gap = 0.045
                frame_fontsize = 11

            fig = plt.figure(figsize=(fig_w, fig_h), dpi=120, facecolor="white")

            # Set up the initial plot and color bar
            one_seq_img = seq_frames[i]
            j = j_indexes[0]
            one_sample_img = one_seq_img[:, :, :, j].cpu()
            rho = torch.squeeze(one_sample_img[0:1, :, :], axis=0)
            mu_v = torch.squeeze(one_sample_img[1:3, :, :], axis=0)

            # General setup for layout
            left = 0.07
            axes_width = 0.80
            data_aspect = float(rho.shape[-1]) / float(rho.shape[-2])

            axes_height = axes_width * fig_w / (data_aspect * fig_h)

            # Reduce the plot width only when a tall dataset would collide with its footer.
            max_axes_height = axes_top - min_axes_bottom
            if axes_height > max_axes_height:
                axes_height = max_axes_height
                axes_width = axes_height * data_aspect * fig_h / fig_w

            # Crucially: calculate bottom after determining the final plot height.
            axes_bottom = axes_top - axes_height
            # Put the footer directly below the grid, rather than at the bottom of the whole GIF canvas.
            frame_text_y = axes_bottom - footer_gap

            ax = fig.add_axes([left, axes_bottom, axes_width, axes_height])
            ax.set_aspect("equal", adjustable="box")
            #ax.set_anchor("C")

            # A dedicated colorbar axes prevents fig.colorbar(..., ax=ax) from resizing
            # the main plot again.
            cbar_gap = 0.016
            cbar_width = 0.020
            cax = fig.add_axes([left + axes_width + cbar_gap, axes_bottom, cbar_width, axes_height])
            # Initial plot and color bar
            axp = ax.matshow(rho, cmap=plt.cm.Blues, vmin=rho_min, vmax=rho_max)
            Q = ax.quiver(mu_v[0], -mu_v[1],color="green",angles="xy",scale_units="xy",scale=self.velScale,minshaft=3.5,width=0.009,headwidth=self.headwidth)
            # color bar setup
            cbar = fig.colorbar(axp, cax=cax, orientation="vertical")
            cbar.set_label("Density rho", fontsize=11)
            cbar.ax.tick_params(labelsize=10)

            # Figure coordinates keep title and animation text independent of axes size.
            fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=13)
            frame_text = fig.text(0.5, frame_text_y, "", ha="center", va="bottom", fontsize=frame_fontsize, fontweight=None if show_metrics_bottom else "bold")

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
                    mask_psnr_text = ""
                    psnr_text      = ""
                    ssim_text      = ""
                    tv_text        = ""
                else:
                    seq_idx = i // 2
                    psnr_text = (f'psnr_rho:{seq_psnr[seq_idx, frame, 0]:.3f}, '
                                 f'psnr_vx:{seq_psnr[seq_idx, frame, 1]:.3f}, '
                                 f'psnr_vy:{seq_psnr[seq_idx, frame, 2]:.3f}'
                                )
                    mask_psnr_text = (f'mpsnr_rho:{seq_masked_psnr[seq_idx, frame, 0]:.3f}, '
                                 f'mpsnr_vx:{seq_masked_psnr[seq_idx, frame, 1]:.3f}, '
                                 f'mpsnr_vy:{seq_masked_psnr[seq_idx, frame, 2]:.3f}'
                                )
                    ssim_text = (f'ssim_rho:{seq_ssim[seq_idx, frame, 0]:.3f}, '
                                 f'ssim_vx:{seq_ssim[seq_idx, frame, 1]:.3f}, '
                                 f'ssim_vy:{seq_ssim[seq_idx, frame, 2]:.3f}'
                                )
                    tv_text   = (f'tv_rho:{seq_tv[seq_idx, frame, 0]:.3f}, '
                                 f'tv_vx:{seq_tv[seq_idx, frame, 1]:.3f}, '
                                 f'tv_vy:{seq_tv[seq_idx, frame, 2]:.3f}'
                                )
                    if frame < self.past_len:
                        frame_text.set_color('black')
                    else:
                        frame_text.set_color('blue')
                if show_metrics_bottom:
                    frame_text.set_text(f'Frame: {frame + 1}/{len(j_indexes)} \n {psnr_text} \n {mask_psnr_text} \n {ssim_text} \n {tv_text}')
                else:
                    frame_text.set_text(f'Frame: {frame + 1}/{len(j_indexes)}')

            # Set up animation for the current sequence
            ani = animation.FuncAnimation(fig, update, frames=len(j_indexes), repeat=True, blit=False)
            # Save each sequence as a separate GIF
            gif_name = f"{self.output_dir}/mprops_GT_seq_{i // 2 + 1}.gif" if (i + 1) % 2 == 0 else f"{self.output_dir}/mprops_seq_{i // 2 + 1}.gif"
            ani.save(gif_name, writer=PillowWriter(fps=2), dpi=120, savefig_kwargs={"facecolor": "white"},)
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

def setup_predictions_plot(predictions, random_past_idx, random_past_samples, random_future_samples, model_fullname, plotType, plotMprop, plotPast, macropropPlotter, show_metrics_bottom=False):
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
    seq_psnr        = get_psnr_per_seq(macropropPlotter.params, pred_seq_list, gt_seq_list, macropropPlotter.eps, masked_flag=False)
    seq_masked_psnr = get_psnr_per_seq(macropropPlotter.params, pred_seq_list, gt_seq_list, macropropPlotter.eps, masked_flag=True)
    seq_ssim = get_ssim_per_seq(macropropPlotter.params, pred_seq_list, gt_seq_list)
    seq_tv   = get_tv_per_seq(pred_seq_list, gt_seq_list, mprops_count=3)

    if plotType == "Static":
        macropropPlotter.plotStatic(seq_frames, match, plotMprop, plotPast)
    elif plotType == "Dynamic":
        macropropPlotter.plotDynamic(seq_frames, seq_psnr, seq_masked_psnr, seq_ssim, seq_tv, show_metrics_bottom)

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

def get_psnr_per_seq(params, pred_seq_list, gt_seq_list, eps, masked_flag=False):
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
            mask = gt_frame[0] > 0.00001           # rho mask, shape (ROWS, COLS)

            if masked_flag:
                psnr_frame_rho = _my_psnr_masked(gt_frame[0], pred_frame[0], rho_range, eps, mask)
                psnr_frame_vx  = _my_psnr_masked(gt_frame[1], pred_frame[1], vx_range,  eps, mask)
                psnr_frame_vy  = _my_psnr_masked(gt_frame[2], pred_frame[2], vy_range,  eps, mask)
            else:
                psnr_frame_rho = _my_psnr(gt_frame[0], pred_frame[0], rho_range, eps)
                psnr_frame_vx  = _my_psnr(gt_frame[1], pred_frame[1], vx_range,  eps)
                psnr_frame_vy  = _my_psnr(gt_frame[2], pred_frame[2], vy_range,  eps)

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

def _compute_tv(field):
    # field shape: (ROWS, COLS)
    diff_rows = np.abs(np.diff(field, axis=0))  # vertical
    diff_cols = np.abs(np.diff(field, axis=1))  # horizontal
    return diff_rows.sum() + diff_cols.sum()

def get_tv_per_seq(pred_seq_list, gt_seq_list, mprops_count):
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    nsamples_tv = np.zeros((nsamples, pred_len, mprops_count))

    for i in range(nsamples):
        one_pred_seq = pred_seq_list[i].cpu().numpy()
        one_gt_seq   = gt_seq_list[i].cpu().numpy()

        for j in range(pred_len):
            for c in range(mprops_count):
                tv_pred = _compute_tv(one_pred_seq[c, :, :, j])
                tv_gt   = _compute_tv(one_gt_seq[c,   :, :, j])
                nsamples_tv[i, j, c] = np.abs(tv_pred - tv_gt)

    return nsamples_tv