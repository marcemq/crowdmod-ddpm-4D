import logging
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
 
from utils.plot.plot_sampled_mprops import FIGSIZE_MAP
 
 
def plot_variability(mean_pred, var_pred, past_seq, seq_idx, output_dir, cfg,
                      velUncScale=3.0, rho_cmap='Blues', var_cmap='inferno'):
    """
    Two-panel figure per predicted frame:
      Left  : mean density (heatmap) + mean velocity (quiver)   -- what the model predicts
      Right : density variance (heatmap) + velocity variability (circles) -- how uncertain it is
 
    Velocity variability is shown the same way this codebase already shows
    sigma2_v (see utils/plot/plot.py::drawMacroProps and the "Uncertainty"
    mode in MacropropPlotter.plotStatic): a circle at each grid cell whose
    radius scales with sqrt(var_vx + var_vy). This reuses an existing visual
    convention instead of introducing a new one.
 
    Args:
        mean_pred: (n_past_seqs, C, ROWS, COLS, F) tensor from compute_variability
        var_pred:  (n_past_seqs, C, ROWS, COLS, F) tensor from compute_variability
        past_seq:  (n_past_seqs, C, ROWS, COLS, PAST_LEN) tensor, for the rho color scale
        seq_idx:   which past sequence (row) to plot
        output_dir: where to save figures
        cfg: run config, used for dataset name / figsize / future_len
    """
    dataset_name = cfg.DATASET.NAME
    figsize = FIGSIZE_MAP.get(dataset_name, (10, 5))
    future_len = cfg.DATASET.FUTURE_LEN
    rows, cols = cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS
 
    mean_seq = mean_pred[seq_idx].cpu().numpy()   # (C, ROWS, COLS, F)
    var_seq  = var_pred[seq_idx].cpu().numpy()    # (C, ROWS, COLS, F)
 
    rho_max = mean_seq[0].max()
 
    for j in range(future_len):
        mean_rho = mean_seq[0, :, :, j]
        mean_vx  = mean_seq[1, :, :, j]
        mean_vy  = mean_seq[2, :, :, j]
 
        var_rho  = var_seq[0, :, :, j]
        var_vx   = var_seq[1, :, :, j]
        var_vy   = var_seq[2, :, :, j]
        vel_variability = np.sqrt(var_vx + var_vy)   # per-cell velocity "spread"
 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]), facecolor='white')
 
        # --- Left: mean prediction ---
        axp1 = ax1.matshow(mean_rho, cmap=plt.cm.get_cmap(rho_cmap), vmin=0, vmax=rho_max)
        ax1.quiver(mean_vx, -mean_vy, color='green', angles='xy', scale_units='xy',
                   scale=0.5, minshaft=3.5, width=0.009, headwidth=5)
        ax1.set_title(f"Mean prediction (frame f+{j+1})", fontsize=11)
        ax1.axis('off')
        cbar1 = fig.colorbar(axp1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Density rho (mean)', fontsize=9)
 
        # --- Right: variability ---
        axp2 = ax2.matshow(var_rho, cmap=plt.cm.get_cmap(var_cmap))
        x, y = np.mgrid[0:cols, 0:rows]
        for i in range(rows):
            for c in range(cols):
                center = (x[c, i] + mean_vx[i, c], y[c, i] - mean_vy[i, c])
                radius = velUncScale * vel_variability[i, c]
                if radius > 1e-6:
                    circle = plt.Circle(center, radius, fill=False, color='cyan', lw=0.8)
                    ax2.add_artist(circle)
        ax2.set_title(f"Variability across repeats (frame f+{j+1})", fontsize=11)
        ax2.axis('off')
        cbar2 = fig.colorbar(axp2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Density variance', fontsize=9)
 
        fig.suptitle(f"Prediction variability | seq {seq_idx+1} | {dataset_name}", y=1.02, fontsize=12)
        fig.tight_layout()
 
        fig_name = f"{output_dir}/variability_seq{seq_idx+1}_f{j+1}.png"
        fig.savefig(fig_name, bbox_inches='tight', dpi=200)
        plt.close(fig)
        logging.info(f"Saved {fig_name}")
 
 
def plot_variability_summary(var_pred, output_dir, cfg):
    """
    Boxplot-style summary: distribution of per-cell variance (flattened over
    ROWS x COLS) per channel, per predicted frame, across all analyzed past
    sequences. Complements the spatial plots with an aggregate view, similar
    in spirit to the existing metrics boxplots (createBoxPlot).
    """
 
    var_np = var_pred.cpu().numpy()  # (n_past_seqs, C, ROWS, COLS, F)
    n_seqs, C, R, Cc, F = var_np.shape
    labels = ['rho', 'vx', 'vy']
 
    fig, axes = plt.subplots(1, F, figsize=(4 * F, 4), sharey=False)
    if F == 1:
        axes = [axes]
 
    for j in range(F):
        data = {labels[c]: var_np[:, c, :, :, j].flatten() for c in range(C)}
        df = pd.DataFrame(data)
        df.boxplot(ax=axes[j])
        axes[j].set_title(f"frame f+{j+1}")
        axes[j].set_ylabel("Per-cell variance")
 
    fig.suptitle("Distribution of prediction variance across grid cells", y=1.03)
    fig.tight_layout()
    save_path = f"{output_dir}/variability_summary_boxplot.png"
    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    logging.info(f"Saved {save_path}")