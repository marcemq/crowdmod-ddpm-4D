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

def createBoxPlotCollapsed(df, title, columns_to_plot, save_path=None, y_limit=5):
    fig, (ax_main, ax_outlier) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), gridspec_kw={'height_ratios': [4, 1]})

    # Plot boxplots
    box = df[columns_to_plot].boxplot(ax=ax_main)
    df[columns_to_plot].boxplot(ax=ax_outlier)

    # Limit y-axis on the main plot to focus on interesting data
    ax_main.set_ylim(-0.1, y_limit)
    ax_outlier.set_ylim(df[columns_to_plot].max().max() * 0.9, df[columns_to_plot].max().max() * 1.01)

    # Hide the spines between axes
    ax_main.spines['bottom'].set_visible(False)
    ax_outlier.spines['top'].set_visible(False)

    # Diagonal lines to indicate axis break
    d = .5
    kwargs = dict(transform=ax_main.transAxes, color='k', clip_on=False)
    ax_main.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax_main.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax_outlier.transAxes)  # switch to the bottom axes
    ax_outlier.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax_outlier.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # Remove ticks for outlier zone
    ax_outlier.set_yticks([])

    # Set titles and labels
    ax_main.set_title(title, fontsize=14)
    ax_main.set_ylabel('Values')

    # Count and annotate outliers
    for i, col in enumerate(columns_to_plot):
        col_data = df[col]
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        outliers = (col_data > upper_bound).sum()
        ax_main.text(i + 1, y_limit, f"{outliers} outliers", ha='center', va='bottom', fontsize=9, color='red')

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Boxplot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def createBoxPlotWithOutlierInfo(df, title, columns_to_plot, save_path=None, ytick_step=2):
    data = [df[col].dropna().values for col in columns_to_plot]  # Get data as list of arrays
    fig, ax = plt.subplots()

    # Plot boxplot without fliers (outliers)
    box = ax.boxplot(data, patch_artist=True, showfliers=False)

    outlier_counts = []

    # Manually compute outliers per column for annotation
    for i, col_data in enumerate(data):
        q1 = np.percentile(col_data, 25)
        q3 = np.percentile(col_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        outlier_counts.append(len(outliers))

        # Add marker and text
        ax.text(i + 1, upper_bound, f"{len(outliers)} outliers", ha='center', va='bottom', fontsize=9, color='red', rotation=90)

    # Title and labels
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Values")
    ax.set_xticks(range(1, len(columns_to_plot)+1))
    ax.set_xticklabels(columns_to_plot, rotation=45)

    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set consistent y-axis ticks
    y_min = df[columns_to_plot].min().min()
    y_max = df[columns_to_plot].max().max()
    if ytick_step is not None:
        yticks = np.arange(y_min // ytick_step * ytick_step, (y_max // ytick_step + 1) * ytick_step, ytick_step)
        ax.set_yticks(yticks)

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Boxplot saved to {save_path}")
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
        createBoxPlotCollapsed(overall_df, title=title, columns_to_plot=overall_df.columns.tolist(), save_path=save_path)
        #createBoxPlotWithOutlierInfo(overall_df, title=title, columns_to_plot=overall_df.columns.tolist(), save_path=save_path, ytick_step=ytick_step)
    else:
        createBoxPlot(overall_df, title=title, columns_to_plot=overall_df.columns.tolist(), save_path=save_path, ytick_step=ytick_step)