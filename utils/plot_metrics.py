import pandas as pd
from matplotlib import pyplot as plt

def createBoxPlot(df, title, columns_to_plot=None, save_path=None):
    if columns_to_plot:
        df[columns_to_plot].boxplot()
    else:
        df[df.columns.tolist()].boxplot()
    plt.title(title, fontsize=16)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.ylabel('Values')
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Boxplot saved to {save_path}")
    else:
        plt.show()

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

def merge_and_plot_boxplot(df_max, df, title, save_path):
  df_max = df_max.add_prefix('max-')
  # Concatenate both DataFrames side by side
  overall_df = pd.concat([df, df_max], axis=1)
  # Get interleaved column names to reorder the columns
  reordered_columns = [item for pair in zip(df.columns, df_max.columns) for item in pair]
  # Reorder the DataFrame with the interleaved columns
  overall_df = overall_df[reordered_columns]
  createBoxPlot(overall_df, title=title, columns_to_plot=overall_df.columns.tolist(), save_path=save_path)