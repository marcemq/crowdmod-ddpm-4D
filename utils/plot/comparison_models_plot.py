import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
from pathlib import Path
from utils.utils import create_directory
from utils.plot.plot_helpers import model_sort_key, pad_labels

variables  = ['rho', 'vx', 'vy']
var_labels = [r'$\rho$ (rho)', 'vx', 'vy']
frame_cols = ['f6', 'f7', 'f8']
frame_labels = ['f+1', 'f+2', 'f+3']
x = np.arange(len(frame_labels))

# Monospace => every (padded) label has exactly the same pixel width
LABEL_FONT = dict(family='DejaVu Sans Mono', weight='bold')

color_palette = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff',
    '#000000', '#e6beff', '#ff4500', '#00ced1', '#ff1493',
    '#7fff00', '#dc143c', '#00bfff', '#ff8c00', '#adff2f',
]

def build_colors(files: dict) -> dict:
    """
    Build { long_name: color } following model_sort_key order
    (family -> weight type -> integrator -> steps ascending).
    """
    model_keys = sorted(next(iter(files.values())).keys(), key=model_sort_key)
    return {
        long_name: color_palette[i % len(color_palette)]
        for i, long_name in enumerate(model_keys)
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

    keys = {
        'psnr_otime':      "PSNR_OVER_TIME",
        'mpsnr_otime':     "MASK_PSNR_OVER_TIME",
        'ssim_otime':      "SSIM_OVER_TIME",
        'tv_otime':        "TV_OVER_TIME",
        'max_psnr_otime':  "MAX_PSNR_OVER_TIME",
        'max_mpsnr_otime': "MAX_MASK_PSNR_OVER_TIME",
        'max_ssim_otime':  "MAX_SSIM_OVER_TIME",
        'psnr':            "PSNR",
        'mpsnr':           "MASK_PSNR",
        'ssim':            "SSIM",
        'max_psnr':        "MAX_PSNR",
        'max_mpsnr':       "MAX_MASK_PSNR",
        'max_ssim':        "MAX_SSIM",
        'bhatt':           "MF_BHATT_COEF",
    }
    out = {k: {} for k in keys}

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        metrics_json_file = model_dir / "metrics_files.json"
        if not metrics_json_file.exists():
            continue

        with open(metrics_json_file) as f:
            m = json.load(f)

        label = model_dir.name.replace('_mE000', '')
        for k, json_key in keys.items():
            out[k][label] = resolve_path(base, m[json_key])

    return out

def draw_row_labels(ax, colors, labels):
    """Draw the model names as an aligned, equal-width column left of the axes."""
    ax.set_yticklabels([])
    for mi, (model, color) in enumerate(colors.items()):
        ax.text(-0.02, mi, labels[model],
                transform=ax.get_yaxis_transform(),
                ha='right', va='center',
                fontsize=9, color=color,
                family=LABEL_FONT['family'], weight=LABEL_FONT['weight'])

def summary_figsize(n_models, n_cols):
    """Grow the figure height with the number of models so rows don't collide."""
    width = 7 if n_cols == 3 else 5
    height = max(3.5, 0.32 * n_models)
    return (width, height)

def metrics_comparison_models(title, files_dict, figure_name, ylim, colors):
    labels = pad_labels(colors.keys())

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
                x + offset,
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
                label=labels[model],
            )

        ax.set_title(var_label, fontsize=12, fontweight='medium', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(frame_labels, fontsize=9)
        ax.set_xlabel('Predicted frame', fontsize=9, color='#888888')
        ax.set_xlim(-0.4, len(frame_labels) - 0.6)
        ax.set_ylim(ylim)
        ax.tick_params(axis='y', labelsize=9, colors='#888888')
        ax.tick_params(axis='x', colors='#888888')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_edgecolor('#cccccc')
        ax.yaxis.grid(True, color='#eeeeee', zorder=0)
        ax.set_axisbelow(True)

    legend_handles = [
        mlines.Line2D([], [], color=color, linewidth=0.5, linestyle='-', marker='o',
                      markersize=3, markerfacecolor=color, markeredgecolor=color,
                      label=labels[model])
        for model, color in colors.items()
    ]

    fig.suptitle(title, fontsize=13, fontweight='medium', y=0.95)
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=4,
        prop={'family': LABEL_FONT['family'], 'size': 8},
        frameon=False,
        bbox_to_anchor=(0.5, -0.15),
    )

    plt.tight_layout()
    plt.savefig(figure_name + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(figure_name + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def _summary_row(ax, y, color, med, q1, q3, value_fontsize=6.5, value_offset=0.2):
    ax.hlines(y, q1, q3, color=color, linewidth=1.5)
    ax.vlines([q1, q3], y - 0.15, y + 0.15, color=color, linewidth=1.0)
    ax.plot(med, y, 'o', color=color, markersize=5, zorder=3)
    # y axis is inverted (first model on top), so y - offset is visually *above* the marker
    ax.text(med, y - value_offset, f'{med:.2f}',
            ha='center', va='bottom', fontsize=value_fontsize,
            color=color, fontweight='bold')

def _style_summary_axis(ax, pi, n_models, y_positions, xlim):
    ax.set_yticks(y_positions)
    # inverted limits: model 0 is drawn at the top, so reading order = sort order
    ax.set_ylim(n_models - 0.4, -0.6)
    if xlim:
        ax.set_xlim(xlim)
    ax.tick_params(axis='x', labelsize=9, colors='#888888')
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_edgecolor('#cccccc')
    ax.xaxis.grid(True, color='#eeeeee', zorder=0)
    ax.set_axisbelow(True)

def metrics_summary(title, files_dict, figure_name, ylabel, colors, xlim=None, files_max_dict=None, value_fontsize=7.5):
    labels = pad_labels(colors.keys())

    stats = {}
    for name, path in files_dict.items():
        df = pd.read_csv(path)
        stats[name] = {
            var: {'med': df[var].median(),
                  'q1':  df[var].quantile(0.25),
                  'q3':  df[var].quantile(0.75)}
            for var in variables
        }
    max_stats = {}
    if files_max_dict:
        for name, path in files_max_dict.items():
            df = pd.read_csv(path)
            max_stats[name] = {var: df[var].median() for var in variables}

    model_names = list(colors.keys())
    y_positions = np.arange(len(model_names))

    fig, axes = plt.subplots(1, 3, figsize=summary_figsize(len(model_names), 3), sharey=True)
    fig.subplots_adjust(wspace=0.15)

    for pi, (var, var_label) in enumerate(zip(variables, var_labels)):
        ax = axes[pi]

        for mi, (model, color) in enumerate(colors.items()):
            s = stats[model][var]
            _summary_row(ax, y_positions[mi], color, s['med'], s['q1'], s['q3'], value_fontsize=value_fontsize)

            if files_max_dict:
                max_med = max_stats[model][var]
                ax.plot(max_med, y_positions[mi], 'o', color=color, markersize=5,
                        markerfacecolor='white', markeredgewidth=1.0,
                        markeredgecolor=color, zorder=3)
                ax.hlines(y_positions[mi], s['med'], max_med, color=color,
                          linewidth=0.8, linestyle='--', alpha=0.5)

        ax.set_title(var_label, fontsize=13, fontweight='medium', pad=8)
        ax.set_xlabel(ylabel, fontsize=10, color='#888888')
        _style_summary_axis(ax, pi, len(model_names), y_positions, xlim)

        if pi == 0:
            draw_row_labels(ax, colors, labels)

    fig.suptitle(title, fontsize=13, fontweight='medium', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15)
    plt.savefig(figure_name + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(figure_name + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def bathh_comparison_models(title, files_dict, figure_name, colors, xlim=None):
    labels = pad_labels(colors.keys())
    bhatt_variables  = ['BHATT_COEF_Hist_2D_Based', 'BHATT_COEF_Hist_1D_Based']
    bhatt_var_labels = ['BHATT_COEF_Hist_2D', 'BHATT_COEF_Hist_1D']

    stats = {}
    for name, path in files_dict.items():
        df = pd.read_csv(path)
        stats[name] = {
            var: {'med': df[var].median(),
                  'q1':  df[var].quantile(0.25),
                  'q3':  df[var].quantile(0.75)}
            for var in bhatt_variables
        }

    model_names = list(colors.keys())
    y_positions = np.arange(len(model_names))

    fig, axes = plt.subplots(1, 2, figsize=summary_figsize(len(model_names), 2), sharey=True)
    fig.subplots_adjust(wspace=0.15)

    for pi, (bhatt_var, bhatt_var_label) in enumerate(zip(bhatt_variables, bhatt_var_labels)):
        ax = axes[pi]

        for mi, (model, color) in enumerate(colors.items()):
            s = stats[model][bhatt_var]
            _summary_row(ax, y_positions[mi], color, s['med'], s['q1'], s['q3'])

        ax.set_title(bhatt_var_label, fontsize=13, fontweight='medium', pad=8)
        _style_summary_axis(ax, pi, len(model_names), y_positions, xlim)

        if pi == 0:
            draw_row_labels(ax, colors, labels)

    fig.suptitle(title, fontsize=13, fontweight='medium', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15)
    plt.savefig(figure_name + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(figure_name + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to create the comparison plots")
    parser.add_argument('--dataset', type=str, default='HERMES-BO', help='Specific dataset name, options: ATC|HERMES-T|HERMES-BO|HERMES-BN|HERMES-CR-90|HERMES-HERMES-CR-90-OBST')
    parser.add_argument('--main-arch', type=str, default='DDPM-UNet', help='Main architecture, options: DDPM-UNet|FM-UNet|ConvRNN')
    parser.add_argument('--raw-metrics-dir', type=str, default='output_hermes_bo/',help='Raw metrics directory')
    args = parser.parse_args()
    files = load_files_dicts(args.raw_metrics_dir)
    colors = build_colors(files)
    labels = pad_labels(colors.keys())

    print("Discovered models (plot order, top to bottom):")
    for long_name, color in colors.items():
        print(f"  {long_name:55s} -> {labels[long_name]}  ({color})")

    out_dir = Path(args.raw_metrics_dir) / "comp_plots"
    create_directory(out_dir)
    tv_range = (0, 80) if args.main_arch=="DDPM-UNet" else (0, 25)

    short_ds_names = {
        "ATC":             "atc",
        "HERMES-T":        "t",
        "HERMES-BO":       "bo",
        "HERMES-BN":       "bn",
        "HERMES-CR-90":    "cr_90",
        "HERMES-CR-90-OBST": "cr_90_obst",
    }

    plots_config_otime = {
        'psnr_otime':     ('PSNR',          files['psnr_otime'],      (10, 42)),
        'mpsnr_otime':    ('MASK_PSNR',     files['mpsnr_otime'],     (10, 42)),
        'ssim_otime':     ('SSIM',          files['ssim_otime'],      (0, 1)),
        'tv_otime':       ('TV',            files['tv_otime'],        tv_range),
        'max_psnr_otime': ('MAX_PSNR',      files['max_psnr_otime'],  (10, 42)),
        'max_mpsnr_otime':('MAX_MASK_PSNR', files['max_mpsnr_otime'], (10, 42)),
        'max_ssim_otime': ('MAX_SSIM',      files['max_ssim_otime'],  (0, 1)),
    }

    plots_config = {
        'psnr':     ('PSNR',          files['psnr'],      (10, 40)),
        'mpsnr':    ('MASK_PSNR',     files['mpsnr'],     (10, 40)),
        'ssim':     ('SSIM',          files['ssim'],      (0.2, 1)),
        'max_psnr': ('MAX_PSNR',      files['max_psnr'],  (10, 40)),
        'max_mpsnr':('MAX_MASK_PSNR', files['max_mpsnr'], (10, 40)),
        'max_ssim': ('MAX_SSIM',      files['max_ssim'],  (0.2, 1)),
    }

    for key, (metric_label, files_dict, ylim) in plots_config_otime.items():
        metrics_comparison_models(
            title=f'{args.dataset} -- {metric_label} over predicted frames',
            files_dict=files_dict,
            figure_name=str(out_dir / f'{key}_{short_ds_names[args.dataset]}'),
            ylim=ylim,
            colors=colors
        )

    for key, (metric_label, files_dict, xlim) in plots_config.items():
        metrics_summary(
            title=f'{args.dataset} -- {metric_label} summary',
            files_dict=files_dict,
            figure_name=str(out_dir / f'summary_{key}_{short_ds_names[args.dataset]}'),
            ylabel=metric_label,
            xlim=xlim,
            colors=colors,
        )

    bathh_comparison_models(title=f"{args.dataset} -- BHATT COEF of motion feature summary",
                    files_dict=files['bhatt'],
                    figure_name=str(out_dir / f"summary_bhatt_{short_ds_names[args.dataset]}"),
                    xlim=(0.2, 0.8),
                    colors=colors,
                )