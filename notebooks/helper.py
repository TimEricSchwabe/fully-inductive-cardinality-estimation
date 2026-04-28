import os
from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Literal



# Define constants that are referenced in the function
# These should be defined at the module level
FIGSIZE_DOUBLE = (16, 8)
METHODS_MAP = {}
METHODS_TO_PLOT = []
METHOD_STYLES = {}
SAVE_KWARGS = {'bbox_inches': 'tight', 'dpi': 300}

# Set publication quality plot parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 22,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


# def plot_boxplot_per_size(df: pd.DataFrame, output_dir: str):
#     """Boxplot per query size (true cost)"""
#     real_cols = [c for c in df.columns if '_real' in c]
#     if not real_cols:
#         return
    
#     melted = df.melt(id_vars=['query_size'], value_vars=real_cols, var_name='Method', value_name='Cost')
#     melted['Method'] = melted['Method'].str.replace('_real', '')
#     melted = melted.dropna(subset=['Cost'])
#     if melted.empty:
#         return
    
#     fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    
#     try:
#         method_ranks = {METHODS_MAP[k]: i for i, k in enumerate(METHODS_TO_PLOT) if k in METHODS_MAP}
#         unique_methods = melted['Method'].unique()
#         hue_order = sorted(unique_methods, key=lambda m: method_ranks.get(m, 99))
        
#         palette = {m: METHOD_STYLES.get(m, {}).get('color') for m in hue_order}
#         missing = [m for m, c in palette.items() if c is None]
#         if missing:
#             fallback_colors = sns.color_palette("colorblind", n_colors=len(missing))
#             for m, c in zip(missing, fallback_colors):
#                 palette[m] = c
        
#         sns.boxplot(data=melted, x='query_size', y='Cost', hue='Method',
#                     palette=palette, hue_order=hue_order, ax=ax,
#                     linewidth=0.6, showfliers=False)
        
#         # Add labels vertically above whiskers
#         unique_sizes = sorted(melted['query_size'].unique())
#         num_hues = len(hue_order)
#         width = 0.8
#         hue_width = width / num_hues
        
#         for i, size in enumerate(unique_sizes):
#             for j, method in enumerate(hue_order):
#                 subset = melted[(melted['query_size'] == size) & (melted['Method'] == method)]
#                 if subset.empty:
#                     continue
                
#                 costs = subset['Cost']
#                 if costs.empty:
#                     continue
                
#                 q1 = costs.quantile(0.25)
#                 q3 = costs.quantile(0.75)
#                 iqr = q3 - q1
#                 upper_lim = q3 + 1.5 * iqr
#                 whisker_val = costs[costs <= upper_lim].max()
#                 if pd.isna(whisker_val):
#                     whisker_val = costs.max()
                
#                 # Calculate x position
#                 # x-tick is at i
#                 x_pos = i + (j - num_hues/2 + 0.5) * hue_width
                
#                 style = METHOD_STYLES.get(method, {})
#                 color = style.get('color', 'black')
                
#                 ax.text(x_pos, whisker_val * 1.25, method,
#                         rotation=90, ha='center', va='bottom',
#                         fontsize=5, color=color)
        
#         ax.legend(loc='best', ncol=3, fontsize=5, frameon=True,
#                   handlelength=0.8, columnspacing=0.4)
#     except ImportError:
#         melted.boxplot(column='Cost', by=['query_size', 'Method'], rot=45, ax=ax, showfliers=False)
    
#     ax.set_yscale('log')
#     ax.set_ylabel('True Cost')
#     ax.set_xlabel('Query Size (Triple Patterns)')
    
#     fig.savefig(os.path.join(output_dir, 'boxplot_per_size_true.pdf'), **SAVE_KWARGS)
#     plt.close(fig)


def q_error(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Q-error between ground truth and predictions."""
    epsilon = 1e-7   
    gt = np.maximum(gt, epsilon)
    pred = np.maximum(pred, epsilon)
    return np.maximum(gt / pred, pred / gt)

def mse_loss(gt: np.ndarray, pred: np.ndarray, ax: int = None) -> float:
    mse = (np.square(gt - pred))
    return mse

        


def process_gcare(path: Union[Path, str]):
    if isinstance(path, str):
        path = Path(path)

    df = pd.DataFrame()
    for method in ["cset", "wj"]:
        base_path = path / method
        gts = np.load(base_path / f"gts.npy")
        preds = np.load(base_path / f"preds.npy")

        q_errors = q_error(gts, preds)
        temp_df = pd.DataFrame({
            "method": method,
            "true_value": gts,
            "prediction": preds,
            "q_error": q_errors
        })

        df = pd.concat([df, temp_df], ignore_index=True)
    return df

def process_gnce(path: Union[Path, str], dataset: str, type: str , epoch: int = None):
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"GNCE predictions file not found at {path}")
    
    if path.is_dir():
        path = path / "final_predictions.csv"
    
    data = pd.read_csv(path)
    data['method'] = 'gnce'
    data = data[data['DATASET'].str.endswith(type) & (data['DATASET'].str.contains(dataset))]
    data['q_error'] = q_error(data['GT'].to_numpy(), data['PREDICTION'].to_numpy())
    data = data.rename(columns={'GT': 'true_value', 'PREDICTION': 'prediction'})
    data = data[['q_error', 'prediction', 'true_value', 'method']]

    return data



def process_lmkg(path: Union[Path, str], lmkg_name: str, q_type: str):
    if isinstance(path, str):
        path = Path(path)
    pattern = f"*mega_no_{lmkg_name}_{q_type}*.csv"
    matches = list(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No LMKG file found matching {pattern} in {path}")
    data = pd.read_csv(matches[0])
    data['q_error'] = q_error(data['true_value'].to_numpy(), data['prediction'].to_numpy())
    data = data[['q_error', 'prediction', 'true_value']]
    data['method'] = 'lmkgs'
    return data


def process_fice(path: Union[Path, str], dataset: str, type: str , epoch: int = None):
    if isinstance(path, str):
        path = Path(path)

    metric: Literal['loss', 'q_error'] = "loss"
    
    if "logs" not in [p.name for p in path.iterdir()]:
        intermedia_step_path = list(path.glob("run_*"))[0]
    else:
        intermedia_step_path = path

    data_path = intermedia_step_path / "logs"

    data = pd.read_csv(data_path / "predictions.csv")
    data = data[(data["dataset_name"] == dataset) & (data['query_type'] == type) & (data['stage'] == 'test')]
    data = data[['epoch', 'prediction', 'true_value']]
    data['q_error'] = q_error(data['true_value'].to_numpy(), data['prediction'].to_numpy())
    data['loss'] = mse_loss(data['true_value'].to_numpy(), data['prediction'].to_numpy())

    if epoch is None:
        if metric == 'q_error':
            print("Selecting best epoch based on Q-Error")
            data_grouped_by_epoch = data.groupby('epoch')['q_error'].mean()
        elif metric == 'loss':
            print("Selecting best epoch based on Loss")
            data_grouped_by_epoch = data.groupby('epoch')['loss'].mean()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        best_epoch = data_grouped_by_epoch.idxmin()
        data = data[data['epoch'] == best_epoch]
        print(f"Using best epoch: {best_epoch}")
    elif epoch == -1:
        print("Selecting last epoch")
        data = data[data['epoch'] == data['epoch'].max()]
    else:
        print(f"Selecting epoch {epoch}")
        data = data[data['epoch'] == epoch]

    data = data[['q_error', 'prediction', 'true_value', 'loss']]
    data["method"] = 'fice'

    return data

transductive_fice_data = None
def process_fice_transductive(path: Union[Path, str], dataset: str, type: str , epoch: int = None):
    global transductive_fice_data
    if isinstance(path, str):
        path = Path(path)

    metric: Literal['loss', 'q_error'] = "loss"
   
    intermedia_step_path = list(path.glob("run_*"))[0]

    data_path = intermedia_step_path / "logs"

    if transductive_fice_data is not None:
        data = transductive_fice_data
    else:
        data = pd.read_csv(data_path / "predictions.csv")
        data = data[(data["stage"] == 'test')]
        data = data[['epoch', 'dataset_name','query_type','prediction', 'true_value']]
        data['q_error'] = q_error(data['true_value'].to_numpy(), data['prediction'].to_numpy())
        data['loss'] = mse_loss(data['true_value'].to_numpy(), data['prediction'].to_numpy())
        transductive_fice_data = data

    data = data[(data["dataset_name"] == dataset) & (data['query_type'] == type)]
    data = data[['epoch', 'prediction', 'true_value', 'q_error', 'loss']]


    if epoch is None:
        if metric == 'q_error':
            print("Selecting best epoch based on Q-Error")
            data_grouped_by_epoch = data.groupby('epoch')['q_error'].mean()
        elif metric == 'loss':
            print("Selecting best epoch based on Loss")
            data_grouped_by_epoch = data.groupby('epoch')['loss'].mean()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        best_epoch = data_grouped_by_epoch.idxmin()
        data = data[data['epoch'] == best_epoch]
        print(f"Using best epoch: {best_epoch}")
    elif epoch == -1:
        print("Selecting last epoch")
        data = data[data['epoch'] == data['epoch'].max()]
    else:
        print(f"Selecting epoch {epoch}")
        data = data[data['epoch'] == epoch]

    data = data[['q_error', 'prediction', 'true_value', 'loss']]
    data["method"] = 'fice_transductive'
    return data

    
    

def plot_q_error_boxplot_per_cardinality(df: pd.DataFrame, output_dir: str = ".", display: bool = False):
    """Boxplot of Q-Error per cardinality bin."""
    if df.empty:
        return

    # Create exactly 5 bins for true_value (cardinality)
    # The last bin captures everything above the 4th threshold
    bins = [0, 1, 10, 100, 1000, np.inf]
    labels = ['0-1', '2-10', '11-100', '101-1000', '>1000']
    
    # Use 'bin' column for x-axis
    df = df.copy()
    df['bin'] = pd.cut(df['true_value'], bins=bins, labels=labels, include_lowest=True)
    
    # Remove rows that fell outside bins (shouldn't happen with np.inf)
    df = df.dropna(subset=['bin'])
    
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    
    try:
        # Determine hue order based on global config if available
        method_ranks = {METHODS_MAP.get(k, k): i for i, k in enumerate(METHODS_TO_PLOT)}
        unique_methods = df['method'].unique()
        # Fallback to current order if method not in ranks
        hue_order = sorted(unique_methods, key=lambda m: method_ranks.get(m, 99))
        
        # Determine palette
        palette = {}
        for m in hue_order:
            style = METHOD_STYLES.get(m, {})
            color = style.get('color')
            palette[m] = color
            
        # Fill missing colors
        missing = [m for m, c in palette.items() if c is None]
        if missing:
            fallback_colors = sns.color_palette("colorblind", n_colors=len(missing))
            for m, c in zip(missing, fallback_colors):
                palette[m] = c
        
        sns.boxplot(data=df, x='bin', y='q_error', hue='method',
                    palette=palette, hue_order=hue_order, ax=ax,
                    linewidth=1.5, showfliers=False)
        
        # Add labels vertically above whiskers
        unique_bins = labels # Use the explicit labels to ensure order
        num_hues = len(hue_order)
        width = 0.8
        hue_width = width / num_hues
        
        for i, bin_label in enumerate(unique_bins):
            for j, method in enumerate(hue_order):
                subset = df[(df['bin'] == bin_label) & (df['method'] == method)]
                if subset.empty:
                    continue
                
                costs = subset['q_error']
                if costs.empty:
                    continue
                
                q1 = costs.quantile(0.25)
                q3 = costs.quantile(0.75)
                iqr = q3 - q1
                upper_lim = q3 + 1.5 * iqr
                whisker_val = costs[costs <= upper_lim].max()
                if pd.isna(whisker_val):
                    whisker_val = costs.max()
                
                # Calculate x position
                x_pos = i + (j - num_hues/2 + 0.5) * hue_width
                
                style = METHOD_STYLES.get(method, {})
                color = style.get('color', 'black')
                
                ax.text(x_pos, whisker_val * 1.25, method,
                        rotation=90, ha='center', va='bottom',
                        fontsize=12, color=color)
        
        ax.legend(loc='best', ncol=3, fontsize=16, frameon=True,
                  handlelength=1.5, columnspacing=0.4)
                  
    except Exception as e:
        print(f"Error plotting boxplot: {e}")
        # Fallback simple plot
        sns.boxplot(data=df, x='bin', y='q_error', hue='method', ax=ax, showfliers=False)
    
    ax.set_yscale('log')
    ax.set_ylabel('Q-Error')
    ax.set_xlabel('True Cardinality Range')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    plt.xticks(rotation=45)
    
    fig.savefig(os.path.join(output_dir, 'boxplot_q_error_per_cardinality.pdf'), **SAVE_KWARGS)
    if display:
        plt.show()
    else:
        plt.close(fig)


def plot_cardinality_scatterplot(df: pd.DataFrame, output_dir: str = ".", display: bool = False, separate_plots: bool = False, max_points: int = 1000):
    """Scatterplot of True Cardinality vs Predicted Cardinality."""
    if df.empty:
        return

    # Downsample per method
    df_sampled = pd.DataFrame()
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        if len(method_df) > max_points:
            method_df = method_df.sample(n=max_points, random_state=42)
        df_sampled = pd.concat([df_sampled, method_df], ignore_index=True)
    
    # Do not filter out non-positive values — use symmetric log scale to plot negatives and zeros
    plot_df = df_sampled.copy()

    unique_methods = plot_df['method'].unique()
    
    # Determine palette
    # Determine hue order based on global config if available
    method_ranks = {METHODS_MAP.get(k, k): i for i, k in enumerate(METHODS_TO_PLOT)}
    # Fallback to current order if method not in ranks
    hue_order = sorted(unique_methods, key=lambda m: method_ranks.get(m, 99))
    
    palette = {}
    for m in hue_order:
        style = METHOD_STYLES.get(m, {})
        color = style.get('color')
        palette[m] = color
        
    # Fill missing colors
    missing = [m for m, c in palette.items() if c is None]
    if missing:
        fallback_colors = sns.color_palette("colorblind", n_colors=len(missing))
        for m, c in zip(missing, fallback_colors):
            palette[m] = c

    if separate_plots:
        num_methods = len(unique_methods)
        fig, axes = plt.subplots(1, num_methods, figsize=(6 * num_methods, 6), sharex=True, sharey=True)
        if num_methods == 1:
            axes = [axes]
        
        # Determine global min/max for shared diagonal
        global_min = min(plot_df['true_value'].min(), plot_df['prediction'].min())
        global_max = max(plot_df['true_value'].max(), plot_df['prediction'].max())

        for ax, method in zip(axes, hue_order):
            subset = plot_df[plot_df['method'] == method]
            if subset.empty:
                continue
            
            sns.scatterplot(data=subset, x='true_value', y='prediction', 
                            color=palette.get(method), ax=ax, alpha=0.5, s=40, label=method)
            
            # Diagonal line
            ax.plot([global_min, global_max], [global_min, global_max], color='red', linestyle='--', linewidth=2)

            # Use symmetric log scale so negatives and zeros are shown
            ax.set_xscale('symlog', linthresh=1)
            ax.set_yscale('symlog', linthresh=1)
            ax.set_title(method)
            ax.set_xlabel('True Cardinality')
            if ax == axes[0]:
                ax.set_ylabel('Predicted Cardinality')
            else:
                ax.set_ylabel('')
            ax.legend(loc='upper left', fontsize=16)
            ax.grid(True, which="both", ls="-", alpha=0.2)

    else:
        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
        try:
            sns.scatterplot(data=plot_df, x='true_value', y='prediction', hue='method', 
                            palette=palette, hue_order=hue_order, ax=ax, alpha=0.5, s=40)
            
            # Diagonal line
            min_val = min(plot_df['true_value'].min(), plot_df['prediction'].min())
            max_val = max(plot_df['true_value'].max(), plot_df['prediction'].max())

            ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal')

            # Use symmetric log scale so negatives and zeros are shown
            ax.set_xscale('symlog', linthresh=1)
            ax.set_yscale('symlog', linthresh=1)
            ax.set_xlabel('True Cardinality')
            ax.set_ylabel('Predicted Cardinality')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
            ax.grid(True, which="both", ls="-", alpha=0.2)
            
        except Exception as e:
            print(f"Error plotting scatterplot: {e}")
            # Fallback
            sns.scatterplot(data=plot_df, x='true_value', y='prediction', hue='method', ax=ax)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, which="both", ls="-", alpha=0.2)
    
    fig.savefig(os.path.join(output_dir, 'scatterplot_cardinality.pdf'), **SAVE_KWARGS)
    if display:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    path = "/path/to/FICE/notebooks/gnce_runs/final_predictions_gnce.csv"
    df = process_gnce(path, "wikidata", "path")
    # mappings = {
    #     'WikiData': ('/path/to/Desktop/fice-ablation-wikidata-gut', '/path/to/FICE/notebooks/fice_runs/wikidata'),
    #     'LUBM': ('/path/to/Desktop/ablation-lubm', '/path/to/FICE/notebooks/fice_runs/lubm'),
    #     'DBPedia100k': ('/path/to/Desktop/ablation-study-dbpedia', '/path/to/FICE/notebooks/fice_runs/dbpedia100k')
    # }

    # for dataset_name, (ablation_path, run_path) in mappings.items():
    #     for dtype in ['path', 'star']:
    #         print(f"Processing {dataset_name} with dtype {dtype}")
            
    #         # ablation_df = process_fice(ablation_path, dataset_name, dtype)
    #         run_df = process_fice(run_path, dataset_name, dtype)