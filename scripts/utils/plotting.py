"""
Plotting utilities for FICE training visualization.

This module contains reusable plotting functions for training metrics,
predictions vs true values, and other visualization needs.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_dense_adj, to_scipy_sparse_matrix
import scipy.sparse as sp
import scipy.sparse.linalg as linalg




def compute_and_plot_laplacian_pe(data, k_vectors=3, num_eigenvalues_to_plot=15):
    """
    1. Plots the 'Scree Plot' (Eigenvalues vs Index) to help choose k.
    2. Visualizes the first 'k_vectors' non-trivial eigenvectors on the graph.
    """
    # --- 1. Math: Compute Laplacian and Eigendecomposition ---
    num_nodes = data.num_nodes
    A = to_dense_adj(data.edge_index)[0]
    D = torch.diag(A.sum(dim=1))
    L = D - A
    
    # Get ALL eigenvalues/vectors (sorted low to high)
    # Note: For huge graphs, use sparse solvers to get only top K, 
    # but for visualization/debugging, dense 'eigh' is safer and easier.
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    
    # Convert to numpy for plotting
    evals_np = eigenvalues.numpy()
    evecs_np = eigenvectors.numpy()

    # --- 2. Visualization Setup ---
    G_nx = to_networkx(data, to_undirected=True)
    pos = nx.kamada_kawai_layout(G_nx) # Good force-directed layout
    
    # Create a figure: 1 row. 
    # First slot = Scree Plot. Remaining slots = Graph Visualizations.
    total_plots = k_vectors + 1
    fig, axes = plt.subplots(1, total_plots, figsize=(4 * total_plots, 4))
    
    # --- 3. Plot 1: The Scree Plot (Eigenvalues) ---
    ax_scree = axes[0]
    
    # Determine how many to plot (don't exceed graph size)
    limit = min(num_nodes, num_eigenvalues_to_plot)
    indices = range(limit)
    values = evals_np[:limit]
    
    # Plot the line
    ax_scree.plot(indices, values, marker='o', linestyle='-', color='black', alpha=0.7)
    
    # Highlight the zero eigenvalue
    ax_scree.scatter([0], [values[0]], color='red', zorder=5, label='$\lambda_0=0$')
    
    # Draw a line showing the user's current 'k' choice
    # We draw it at k+1 because index 0 is trivial.
    # So if user wants k=2 vectors, we are looking at indices 1 and 2.
    cutoff_index = k_vectors 
    ax_scree.axvline(x=cutoff_index, color='blue', linestyle='--', label=f'Cutoff (k={k_vectors})')
    
    ax_scree.set_title("Eigenvalue Spectrum (Scree Plot)")
    ax_scree.set_xlabel("Index ($i$)")
    ax_scree.set_ylabel("Eigenvalue ($\lambda_i$)")
    ax_scree.grid(True, alpha=0.3)
    ax_scree.legend()

    # --- 4. Plot 2..N: The Eigenvectors on the Graph ---
    # We start from index 1 because index 0 is the trivial constant vector
    for i in range(k_vectors):
        ax = axes[i + 1] # Shift by 1 because axes[0] is the scree plot
        
        ev_index = i + 1 
        vec = evecs_np[:, ev_index]
        val = evals_np[ev_index]
        
        nx.draw(
            G_nx, pos, ax=ax,
            node_color=vec, 
            cmap='coolwarm', 
            node_size=200, 
            with_labels=False,
            edge_color='gray',
            width=0.8
        )
        ax.set_title(f"Vector $v_{{{ev_index}}}$\n$\lambda_{{{ev_index}}}$ = {val:.4f}")

    plt.tight_layout()
    plt.savefig(f"laplacian_pe_plot_{k_vectors}.png")
    plt.close()
    print(f"Saved Laplacian PE plot to laplacian_pe_plot_{k_vectors}.png")
    
    return eigenvectors[:, 1:k_vectors+1]


def plot_training_metrics(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (18, 10),
    use_log_scale: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot comprehensive training and testing metrics.
    
    Args:
        df: DataFrame containing training metrics with columns:
            - epoch: Epoch numbers
            - train_loss, test_loss: Training and test losses
            - train_mae, test_mae: Mean Absolute Error
            - train_mse, test_mse: Mean Square Error
            - train_rmse, test_rmse: Root Mean Square Error
            - train_q_error, test_q_error: Q-Error metrics
        figsize: Figure size as (width, height)
        use_log_scale: Whether to use logarithmic scale for y-axes
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    
    Returns:
        The matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Set log scale for all subplots if requested
    if use_log_scale:
        for i in range(1, 6):
            plt.subplot(2, 3, i)
            plt.yscale('log')
    
    # Plot training and test losses
    plt.subplot(2, 3, 1)
    plt.plot(df['epoch'], df['train_loss'], 'b-', label='Training Loss')
    plt.plot(df['epoch'], df['test_loss'], 'r-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    # Plot MAE
    plt.subplot(2, 3, 2)
    plt.plot(df['epoch'], df['train_mae'], 'b-', label='Train MAE')
    plt.plot(df['epoch'], df['test_mae'], 'r-', label='Test MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.grid(True)

    # Plot MSE
    plt.subplot(2, 3, 3)
    plt.plot(df['epoch'], df['train_mse'], 'b-', label='Train MSE')
    plt.plot(df['epoch'], df['test_mse'], 'r-', label='Test MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Mean Square Error')
    plt.legend()
    plt.grid(True)

    # Plot RMSE
    plt.subplot(2, 3, 4)
    plt.plot(df['epoch'], df['train_rmse'], 'b-', label='Train RMSE')
    plt.plot(df['epoch'], df['test_rmse'], 'r-', label='Test RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Root Mean Square Error')
    plt.legend()
    plt.grid(True)

    # Plot Q-Error
    plt.subplot(2, 3, 5)
    plt.plot(df['epoch'], df['train_q_error'], 'b-', label='Train Q-ERROR')
    plt.plot(df['epoch'], df['test_q_error'], 'r-', label='Test Q-ERROR')
    plt.xlabel('Epoch')
    plt.ylabel('Q-Error')
    plt.title('Average Q-Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training metrics plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
        print("Training curves and metrics plotted!")
    
    return fig


def plot_predictions_vs_true_values(
    train_predictions_log: List[List[float]],
    train_true_values_log: List[List[float]],
    test_predictions_log: List[List[float]],
    test_true_values_log: List[List[float]],
    figsize: Tuple[int, int] = (16, 8),
    max_points_per_epoch: int = 50,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create scatter plots showing model predictions vs true cardinality values across all epochs.
    
    Args:
        train_predictions_log: List of prediction lists for each training epoch
        train_true_values_log: List of true value lists for each training epoch
        test_predictions_log: List of prediction lists for each test epoch
        test_true_values_log: List of true value lists for each test epoch
        figsize: Figure size as (width, height)
        max_points_per_epoch: Maximum number of points to plot per epoch
        random_seed: Random seed for reproducible point sampling
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    
    Returns:
        The matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    def get_random_indices(data_length: int, max_points: int) -> List[int]:
        """Helper function to randomly sample indices."""
        if data_length <= max_points:
            return list(range(data_length))
        return np.random.choice(data_length, max_points, replace=False).tolist()
    
    # Sample data for plotting
    train_preds_sampled = []
    train_trues_sampled = []
    if train_predictions_log and train_true_values_log:
        idx = get_random_indices(len(train_predictions_log[0]), max_points_per_epoch)
        train_preds_sampled = [[epoch_preds[i] for i in idx] for epoch_preds in train_predictions_log]
        train_trues_sampled = [[epoch_trues[i] for i in idx] for epoch_trues in train_true_values_log]

    test_preds_sampled = []
    test_trues_sampled = []
    if test_predictions_log and test_true_values_log:
        test_idx = get_random_indices(len(test_predictions_log[0]), max_points_per_epoch)
        test_preds_sampled = [[epoch_preds[i] for i in test_idx] for epoch_preds in test_predictions_log]
        test_trues_sampled = [[epoch_trues[i] for i in test_idx] for epoch_trues in test_true_values_log]

    # Plot train predictions vs true values for all epochs
    plt.subplot(1, 2, 1)
    if train_preds_sampled and train_trues_sampled:
        for epoch_idx, (preds, trues) in enumerate(zip(train_preds_sampled, train_trues_sampled)):
            if len(preds) > 0 and len(trues) > 0:
                plt.scatter(trues, preds, alpha=0.3,
                           s=20, label=f'Epoch {epoch_idx+1}' if epoch_idx % 5 == 0 else "")

        # Add perfect prediction line
        all_train_trues = [val for epoch_trues in train_trues_sampled for val in epoch_trues]
        if all_train_trues:
            min_val, max_val = min(all_train_trues), max(all_train_trues)
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

    plt.xlabel('True Cardinality')
    plt.ylabel('Predicted Cardinality')
    plt.title('Train Predictions vs True Values (All Epochs)')
    plt.grid(True)
    if len(train_preds_sampled) <= 10:
        plt.legend(fontsize=8)

    # Plot test predictions vs true values for all epochs
    plt.subplot(1, 2, 2)
    if test_preds_sampled and test_trues_sampled:
        test_colors = plt.cm.plasma(np.linspace(0, 1, len(test_preds_sampled)))
        for epoch_idx, (preds, trues) in enumerate(zip(test_preds_sampled, test_trues_sampled)):
            if len(preds) > 0 and len(trues) > 0:
                plt.scatter(trues, preds, alpha=0.3, color=test_colors[epoch_idx],
                           s=20, label=f'Epoch {epoch_idx+1}' if epoch_idx % 5 == 0 else "")

        # Add perfect prediction line
        all_test_trues = [val for epoch_trues in test_trues_sampled for val in epoch_trues]
        if all_test_trues:
            min_val, max_val = min(all_test_trues), max(all_test_trues)
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

    plt.xlabel('True Cardinality')
    plt.ylabel('Predicted Cardinality')
    plt.title('Test Predictions vs True Values (All Epochs)')
    plt.grid(True)
    if len(test_preds_sampled) <= 10:
        plt.legend(fontsize=8)

    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions vs True Values plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
        print("Predictions vs True Values across all epochs plotted!")
    
    return fig


def plot_simple_training_curves(
    epochs_range: List[int],
    train_losses: List[float],
    eval_epochs: List[int],
    test_losses: List[float],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot simple training and test loss curves.
    
    Args:
        epochs_range: List of epoch numbers for training data
        train_losses: List of training losses
        eval_epochs: List of epoch numbers for test data
        test_losses: List of test losses
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    
    Returns:
        The matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(eval_epochs, test_losses, 'r-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curves plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
        print("Loss curves plotted!")
    
    return fig


def load_and_plot_training_data(
    csv_path: Union[str, Path],
    use_log_scale: bool = True,
    save_metrics_plot: bool = True,
    show_plots: bool = True
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Convenience function to load training data from CSV and create plots.
    
    Args:
        csv_path: Path to the training curves CSV file
        use_log_scale: Whether to use logarithmic scale for metrics plots
        save_metrics_plot: Whether to save the metrics plot automatically
        show_plots: Whether to display the plots
    
    Returns:
        Tuple of (DataFrame, Figure) containing the loaded data and metrics plot
    """
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded training data from: {csv_path}")
    print(f"Data shape: {df.shape}")
    
    # Determine save path for plot
    save_path = None
    if save_metrics_plot:
        csv_path = Path(csv_path)
        save_path = csv_path.parent / f"{csv_path.stem}_metrics_plot.png"
    
    # Create metrics plot
    fig = plot_training_metrics(
        df=df,
        use_log_scale=use_log_scale,
        save_path=save_path,
        show_plot=show_plots
    )
    
    return df, fig