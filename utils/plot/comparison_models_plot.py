import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
from pathlib import Path
from utils.utils import create_directory

variables  = ['rho', 'vx', 'vy']
var_labels = [r'$\rho$ (rho)', 'vx', 'vy']
frame_cols = ['f6', 'f7', 'f8']
frame_labels = ['f+1', 'f+2', 'f+3']
x = np.arange(len(frame_labels))

colors = {
    'DDPM-UNet_sampDDPM':       '#3266ad',
    'FM-UNet_Linear_intgEuler': '#3a9e75',
    'ConvRNN_GRUCell':  '#c45c3a',
    'ConvRNN_LSTMCell': '#8b5db8',
}

short_model_names = {
    'DDPM-UNet_sampDDPM':       'DIF-UNet_sDDPM',
    'FM-UNet_Linear_intgEuler': 'FM-UNet_LpEi',
    'ConvRNN_GRUCell':  'ConvRNN',
    'ConvRNN_LSTMCell': 'ConvLSTM',
}

def resolve_path(base: Path, json_path: str) -> Path:
    """Strip the leading directory from json_path (e.g. 'output_hermes_bn/...') and prepend base."""
    p = Path(json_path)
    return base / p.relative_to(p.parts[0])

def load_files_dicts(raw_metrics_dir: str) -> dict:
    """
    Scans raw_metrics_dir for model subdirectories, reads each metrics_files.json,
    and returns a dict of dicts grouped by metric type.
    """
    base = Path(raw_metrics_dir)

    files_psnr_otime     = {}
    files_ssim_otime     = {}
    files_max_psnr_otime = {}
    files_max_ssim_otime = {}
    files_psnr           = {}
    files_ssim           = {}
    files_max_psnr       = {}
    files_max_ssim       = {}
    files_bhatt          = {}

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        metrics_json_file = model_dir / "metrics_files.json"
        if not metrics_json_file.exists():
            continue

        with open(metrics_json_file) as f:
            m = json.load(f)

        label = model_dir.name.replace('_modelE000', '')
        files_psnr_otime[label]     = resolve_path(base, m["PSNR_OVER_TIME"])
        files_ssim_otime[label]     = resolve_path(base, m["SSIM_OVER_TIME"])
        files_max_psnr_otime[label] = resolve_path(base, m["MAX_PSNR_OVER_TIME"])
        files_max_ssim_otime[label] = resolve_path(base, m["MAX_SSIM_OVER_TIME"])
        files_psnr[label]           = resolve_path(base, m["PSNR"])
        files_ssim[label]           = resolve_path(base, m["SSIM"])
        files_max_psnr[label]       = resolve_path(base, m["MAX_PSNR"])
        files_max_ssim[label]       = resolve_path(base, m["MAX_SSIM"])
        files_bhatt[label]          = resolve_path(base, m["MF_BHATT_COEF"])

    return {
        'psnr_otime':     files_psnr_otime,
        'ssim_otime':     files_ssim_otime,
        'max_psnr_otime': files_max_psnr_otime,
        'max_ssim_otime': files_max_ssim_otime,
        'psnr':           files_psnr,
        'ssim':           files_ssim,
        'max_psnr':       files_max_psnr,
        'max_ssim':       files_max_ssim,
        'bhatt':          files_bhatt,
    }

def metrics_comparison_models(title, files_dict, figure_name, ylim):
    stats = {}
    for name, path in files_dict.items():
        df = pd.read_csv(path)
        stats[name] = {}
        for var in variables:
            stats[name][var] = {'med': [], 'q1': [], 'q3': []}
            for f in frame_cols:
                col = f"{var}_{f}"
                stats[name][var]['med'].append(df[col].median())
                stats[name][var]['q1'].append(df[col].quantile(0.25))
                stats[name][var]['q3'].append(df[col].quantile(0.75))

    fig, axes = plt.subplots(1, 3, figsize=(7, 3), sharey=False)
    fig.subplots_adjust(wspace=0.3)

    for pi, (var, var_label) in enumerate(zip(variables, var_labels)):
        ax = axes[pi]

        n_models = len(colors)
        dodge_step = 0.05
        offsets = np.linspace(-(n_models-1)/2 * dodge_step, (n_models-1)/2 * dodge_step, n_models)

        for (model, color), offset in zip(colors.items(), offsets):
            med  = np.array(stats[model][var]['med'])
            q1   = np.array(stats[model][var]['q1'])
            q3   = np.array(stats[model][var]['q3'])
            yerr = np.array([med - q1, q3 - med])

            ax.errorbar(
                x + offset,   # Apply offset
                med,
                yerr=yerr,
                fmt='o-',
                color=color,
                linewidth=0.8,
                markersize=3,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.5,
                capsize=4,
                capthick=0.8,
                elinewidth=0.8,
                label=short_model_names[model],
            )

        ax.set_title(var_label, fontsize=13, fontweight='medium', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(frame_labels, fontsize=10)
        ax.set_xlabel('Predicted frame', fontsize=10, color='#888888')
        ax.set_xlim(-0.4, len(frame_labels) - 0.6)
        ax.set_ylim(ylim)
        ax.tick_params(axis='y', labelsize=9, colors='#888888')
        ax.tick_params(axis='x', colors='#888888')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_edgecolor('#cccccc')
        ax.yaxis.grid(True, color='#eeeeee', zorder=0)
        ax.set_axisbelow(True)

    legend_handles = []
    for model, color in colors.items():
        handle = mlines.Line2D(
            [], [],
            color=color,
            linewidth=0.5,
            linestyle='-',
            marker='o',
            markersize=3,
            markerfacecolor=color,
            markeredgecolor=color,
            label=short_model_names[model],
        )
        legend_handles.append(handle)

    fig.suptitle(title, fontsize=15, fontweight='medium', y=1.02)
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=4,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    plt.savefig(figure_name + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(figure_name + '.png', bbox_inches='tight', dpi=300)

def metrics_summary(title, files_dict, figure_name, ylabel, xlim=None, files_max_dict=None):
    stats = {}
    for name, path in files_dict.items():
        df = pd.read_csv(path)
        stats[name] = {}
        for var in variables:
            stats[name][var] = {
                'med': df[var].median(),
                'q1':  df[var].quantile(0.25),
                'q3':  df[var].quantile(0.75),
            }
    # load max stats if provided
    max_stats = {}
    if files_max_dict:
        for name, path in files_max_dict.items():
            df = pd.read_csv(path)
            max_stats[name] = {var: df[var].median() for var in variables}

    model_names = list(colors.keys())
    y_positions = np.arange(len(model_names))

    fig, axes = plt.subplots(1, 3, figsize=(7, 3), sharey=True)
    fig.subplots_adjust(wspace=0.15)

    for pi, (var, var_label) in enumerate(zip(variables, var_labels)):
        ax = axes[pi]

        for mi, (model, color) in enumerate(colors.items()):
            med = stats[model][var]['med']
            q1  = stats[model][var]['q1']
            q3  = stats[model][var]['q3']
            y   = y_positions[mi]

            ax.hlines(y, q1, q3, color=color, linewidth=1.5)
            ax.vlines([q1, q3], y - 0.15, y + 0.15, color=color, linewidth=1.0)
            ax.plot(med, y, 'o', color=color, markersize=5, zorder=3)
            ax.text(med, y + 0.22, f'{med:.2f}',
                    ha='center', va='bottom', fontsize=8,
                    color=color, fontweight='bold')

            if files_max_dict:
                max_med = max_stats[model][var]
                ax.plot(max_med, y, 'o', color=color, markersize=5,
                        markerfacecolor='white', markeredgewidth=1.0,
                        markeredgecolor=color, zorder=3)
                ax.hlines(y, med, max_med, color=color,
                        linewidth=0.8, linestyle='--', alpha=0.5)

        ax.set_title(var_label, fontsize=13, fontweight='medium', pad=8)
        ax.set_xlabel(ylabel, fontsize=10, color='#888888')
        ax.set_yticks(y_positions)
        ax.set_ylim(-0.6, len(model_names) - 0.4)
        if xlim:
            ax.set_xlim(xlim)
        ax.tick_params(axis='x', labelsize=9, colors='#888888')
        ax.tick_params(axis='y', left=False, labelleft=(pi == 0))
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_edgecolor('#cccccc')
        ax.xaxis.grid(True, color='#eeeeee', zorder=0)
        ax.set_axisbelow(True)

        if pi == 0:
            ax.set_yticklabels([])
            for mi, (model, color) in enumerate(colors.items()):
                ax.text(-0.02, mi, short_model_names[model],
                        transform=ax.get_yaxis_transform(),
                        ha='right', va='center',
                        fontsize=9, fontweight='bold', color=color)
        else:
            ax.set_yticklabels([])

    fig.suptitle(title, fontsize=13, fontweight='medium', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15)
    plt.savefig(figure_name + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(figure_name + '.png', bbox_inches='tight', dpi=300)

def bathh_comparison_models(title, files_dict, figure_name, xlim=None):
    bhatt_variables  = ['BHATT_COEF_Hist_2D_Based', 'BHATT_COEF_Hist_1D_Based']
    bhatt_var_labels = ['BHATT_COEF_Hist_2D', 'BHATT_COEF_Hist_1D']

    stats = {}
    for name, path in files_dict.items():
        df = pd.read_csv(path)
        stats[name] = {}
        for var in bhatt_variables:
            stats[name][var] = {
                'med': df[var].median(),
                'q1':  df[var].quantile(0.25),
                'q3':  df[var].quantile(0.75),
            }

    model_names = list(colors.keys())
    y_positions = np.arange(len(model_names))

    fig, axes = plt.subplots(1, 2, figsize=(5, 3), sharey=True)
    fig.subplots_adjust(wspace=0.15)

    for pi, (bhatt_var, bhatt_var_label) in enumerate(zip(bhatt_variables, bhatt_var_labels)):
        ax = axes[pi]

        for mi, (model, color) in enumerate(colors.items()):
            med = stats[model][bhatt_var]['med']
            q1  = stats[model][bhatt_var]['q1']
            q3  = stats[model][bhatt_var]['q3']
            y   = y_positions[mi]

            ax.hlines(y, q1, q3, color=color, linewidth=1.5)
            ax.vlines([q1, q3], y - 0.15, y + 0.15, color=color, linewidth=1.0)
            ax.plot(med, y, 'o', color=color, markersize=5, zorder=3)
            ax.text(med, y + 0.22, f'{med:.2f}',
                    ha='center', va='bottom', fontsize=8,
                    color=color, fontweight='bold')

        ax.set_title(bhatt_var_label, fontsize=13, fontweight='medium', pad=8)
        ax.set_yticks(y_positions)
        ax.set_ylim(-0.6, len(model_names) - 0.4)
        if xlim:
            ax.set_xlim(xlim)
        ax.tick_params(axis='x', labelsize=9, colors='#888888')
        ax.tick_params(axis='y', left=False, labelleft=(pi == 0))
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_edgecolor('#cccccc')
        ax.xaxis.grid(True, color='#eeeeee', zorder=0)
        ax.set_axisbelow(True)

        if pi == 0:
            ax.set_yticklabels([])
            for mi, (model, color) in enumerate(colors.items()):
                ax.text(-0.02, mi, short_model_names[model],
                        transform=ax.get_yaxis_transform(),
                        ha='right', va='center',
                        fontsize=9, fontweight='bold', color=color)
        else:
            ax.set_yticklabels([])

    fig.suptitle(title, fontsize=13, fontweight='medium', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15)
    plt.savefig(figure_name + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(figure_name + '.png', bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to create the comparison plots")
    parser.add_argument('--dataset', type=str, default='HERMES-BO', help='Specific dataset name, options: ATC|HERMES-BO|HERMES-BN|HERMES-CR-90|HERMES-HERMES-CR-90-OBST')
    parser.add_argument('--raw-metrics-dir', type=str, default='output_hermes_bo/',help='Raw metrics directory')
    args = parser.parse_args()
    files = load_files_dicts(args.raw_metrics_dir)

    out_dir = Path(args.raw_metrics_dir) / "comp_plots"
    create_directory(out_dir)

    short_ds_names = {
        "ATC":             "atc",
        "HERMES-BO":       "bo",
        "HERMES-BN":       "bn",
        "HERMES-CR-90":    "cr_90",
        "HERMES-CR-90-OBST": "cr_90_obst",
    }

    plots_config_otime = {
        'psnr_otime':     ('PSNR',     files['psnr_otime'],     (10, 42)),
        'ssim_otime':     ('SSIM',     files['ssim_otime'],     (0, 1)),
        'max_psnr_otime': ('MAX-PSNR', files['max_psnr_otime'], (10, 42)),
        'max_ssim_otime': ('MAX-SSIM', files['max_ssim_otime'], (0, 1)),
    }

    plots_config = {
        'psnr':     ('PSNR',     files['psnr'],     (10, 40)),
        'ssim':     ('SSIM',     files['ssim'],     (0.2, 1)),
        'max_psnr': ('MAX-PSNR', files['max_psnr'], (10, 40)),
        'max_ssim': ('MAX-SSIM', files['max_ssim'], (0.2, 1)),
    }

    for key, (metric_label, files_dict, ylim) in plots_config_otime.items():
        metrics_comparison_models(
            title=f'{args.dataset} -- {metric_label} over predicted frames',
            files_dict=files_dict,
            figure_name=str(out_dir / f'{key}_{short_ds_names[args.dataset]}'),  # e.g. psnr_otime_bo
            ylim=ylim,
        )

    for key, (metric_label, files_dict, xlim) in plots_config.items():
        metrics_summary(
            title=f'{args.dataset} -- {metric_label} summary',
            files_dict=files_dict,
            figure_name=str(out_dir / f'summary_{key}_{short_ds_names[args.dataset]}'),  # e.g. summary_psnr_bo
            ylabel=metric_label,
            xlim=xlim
        )

    bathh_comparison_models(title=f"{args.dataset} -- BHATT COEF of motion feature summary",
                    files_dict=files['bhatt'],
                    figure_name=str(out_dir / f"summary_bhatt_{short_ds_names[args.dataset]}"),
                    xlim=(0.2, 0.8))