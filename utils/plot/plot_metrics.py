import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def createBoxPlot(df, title, columns_to_plot, save_path=None, ytick_step=5):
    df[columns_to_plot].boxplot()
    plt.title(title, fontsize=16)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.ylabel('Values')
    # Set consistent y-axis ticks
    y_min, y_max = df[columns_to_plot].min().min(), df[columns_to_plot].max().max()
    if ytick_step is not None:
        yticks = np.arange(y_min // ytick_step * ytick_step, (y_max // ytick_step + 1) * ytick_step, ytick_step)
        plt.yticks(yticks)
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Boxplot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def createBoxPlotWithOutlierInfo(df, title, columns_to_plot, save_path=None, y_limit=4):
    data = [df[col].dropna().values for col in columns_to_plot]
    y_text_pos = 2.5
    fig, ax = plt.subplots(figsize=(len(columns_to_plot) * 1.3, 6))
    bp = ax.boxplot(data, showfliers=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylim(0, y_limit)
    ax.set_ylabel("Values")
    ax.set_title(title)
    ax.set_xticks(np.arange(1, len(columns_to_plot) + 1))
    ax.set_xticklabels(columns_to_plot, rotation=0)

    # Add horizontal grid lines like pandas
    ax.yaxis.grid(True, alpha=0.7)
    ax.xaxis.grid(True, alpha=0.7)

    # Pandas-style color settings
    line_color = '#1f77b4'       # Light blue (default pandas box and whisker)
    median_color = '#2ca02c'     # Green (default pandas median)
    cap_color = 'black'          # Default cap color in pandas

    for box in bp['boxes']:
        box.set_color(line_color)
        box.set_linewidth(1)

    for whisker in bp['whiskers']:
        whisker.set_color(line_color)
        whisker.set_linewidth(1)

    for cap in bp['caps']:
        cap.set_color(cap_color)
        cap.set_linewidth(1)

    for median in bp['medians']:
        median.set_color(median_color)
        median.set_linewidth(1)

    # Annotate outlier counts
    for i, col in enumerate(columns_to_plot):
        col_data = df[col]
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        outlier_count = (col_data > upper_bound).sum()

        # Position text a little to the right of the box
        ax.text(i + 1.1, y_text_pos, f"{outlier_count} outliers", ha='left', va='top', fontsize=9, rotation=90, color='red')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()

def createBoxPlot_bhatt(df1, df2, title, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df1_cols = df1.columns.tolist()
    df1[df1_cols].boxplot(ax=axes[0])
    axes[0].set_title("Bhatt-Coeficient", fontsize=14)
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Values')

    df2_cols = df2.columns.tolist()
    df2[df2_cols].boxplot(ax=axes[1])
    axes[1].set_title("Bhatt-Distance", fontsize=14)
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].set_ylabel('Values')
    # Add overall title
    plt.suptitle(title, fontsize=16)
    # Adjust layout and show the plot
    plt.tight_layout()
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Boxplot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def merge_and_plot_boxplot(df_max, df, title, save_path, ytick_step, prefix='max-', outliersFlag=False):
    df_max = df_max.add_prefix(prefix)
    # Concatenate both DataFrames side by side
    overall_df = pd.concat([df, df_max], axis=1)
    # Get interleaved column names to reorder the columns
    reordered_columns = [item for pair in zip(df.columns, df_max.columns) for item in pair]
    # Reorder the DataFrame with the interleaved columns
    overall_df = overall_df[reordered_columns]
    if outliersFlag:
        createBoxPlotWithOutlierInfo(overall_df, title=title, columns_to_plot=overall_df.columns.tolist(), save_path=save_path)
    else:
        createBoxPlot(overall_df, title=title, columns_to_plot=overall_df.columns.tolist(), save_path=save_path, ytick_step=ytick_step)

def get_angle_tick_labels(num_angle_bins):
    # Angle ticks depend on num_angle_bins
    step = np.pi / (num_angle_bins // 2)  # e.g. 8 bins → π/4, 16 bins → π/8
    angle_ticks = np.arange(-np.pi, np.pi + step, step)

    # Create labels nicely in terms of π
    def format_pi(x):
        frac = x / np.pi
        if np.isclose(frac, 0): return "0"
        if np.isclose(frac, 1): return r"$\pi$"
        if np.isclose(frac, -1): return r"$-\pi$"
        num, den = (frac).as_integer_ratio()
        if den == 1:
            return fr"${num}\pi$"
        else:
            return fr"${num}\pi/{den}$"

    angle_tick_labels = [format_pi(val) for val in angle_ticks]
    return angle_ticks, angle_tick_labels

def plot_motion_feat_hist2D(hist_2D, mag_edges, angle_edges, sample, i, row, col, plotted_idx, output_dir, num_angle_bins):
    plt.figure(figsize=(5, 4))
    plt.imshow(hist_2D.T, origin='lower', aspect='auto', extent=[mag_edges[0], mag_edges[-1], angle_edges[0], angle_edges[-1]], cmap='viridis')
    plt.colorbar(label="Counts")
    # Magnitude ticks every 1
    plt.xticks(np.arange(mag_edges[0], mag_edges[-1] + 1, 1))

    # Angle ticks at -π, -3π/4, -π/2, ..., π
    angle_ticks, angle_tick_labels = get_angle_tick_labels(num_angle_bins)
    plt.yticks(angle_ticks, angle_tick_labels)

    plt.xlabel("Magnitude bin")
    plt.ylabel("Angle bin (radians)")
    plt.title(f"2D Motion Hist | Sample {sample}, Block ({i},{row},{col})")
    save_path = f"{output_dir}/mf_hist2D_plot{plotted_idx}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_motion_feat_hist1D(hist_1D, sample, i, row, col, num_plot, output_dir, num_angle_bins):
    angle_bin_edges = np.linspace(-np.pi, np.pi, num_angle_bins+1)
    angle_bin_centers = (angle_bin_edges[:-1] + angle_bin_edges[1:]) / 2

    plt.figure(figsize=(5, 4))
    plt.bar(angle_bin_centers, hist_1D, width=(2*np.pi / len(hist_1D)), align='center', alpha=0.7, color='steelblue', edgecolor='black')
    # Set ticks every π/4 or π/8 
    xticks, xtick_labels = get_angle_tick_labels(num_angle_bins)
    plt.xticks(xticks, xtick_labels)

    plt.xlabel("Angle (radians)")
    plt.ylabel("Weighted magnitude sum")
    plt.title(f"1D Motion Hist | Sample {sample}, Block (t={i}, r={row}, c={col})")
    save_path = f"{output_dir}/mf_hist1D_plot{num_plot}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()