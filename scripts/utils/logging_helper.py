"""
Helper functions for logging metrics, saving CSVs, and managing training data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.decomposition import PCA
from wandb.errors import AuthenticationError
import time
from sklearn.manifold import TSNE
import utils.metrics as metrics



logger = logging.getLogger(__name__)


# CSV file name constants
TRAIN_METRICS_FILE_NAME = "train_training_metrics.csv"
TEST_METRICS_FILE_NAME = "test_training_metrics.csv"
TRAIN_METRICS_PER_GRAPH_FILE_NAME = "train_metrics_per_graph.csv"
TEST_METRICS_PER_GRAPH_FILE_NAME = "test_metrics_per_graph.csv"
TRAIN_METRICS_PER_QUERY_TYPE_FILE_NAME = "train_metrics_per_query_type.csv"
TEST_METRICS_PER_QUERY_TYPE_FILE_NAME = "test_metrics_per_query_type.csv"
TRAIN_METRICS_PER_QUERY_SET_NAME_FILE_NAME = "train_metrics_per_query_set_name.csv"
TEST_METRICS_PER_QUERY_SET_NAME_FILE_NAME = "test_metrics_per_query_set_name.csv"
PREDICTIONS_FILE_NAME = "predictions.csv"


# ============================================================================
# Helper Functions for Metrics, Logging, and Data Management
# ============================================================================

def calculate_all_metrics(predictions_log, true_values_log):
    """
    Calculate all metrics from log predictions.
    
    Args:
        predictions_log: Array/list of predictions in log space
        true_values_log: Array/list of true values in log space
        
    Returns:
        Dictionary containing all calculated metrics
    """
    predictions = np.expm1(predictions_log)
    true_values = np.expm1(true_values_log)
    mse = metrics.calculate_mse(predictions, true_values)
    
    epsilon = 1e-7
    p_safe = np.maximum(predictions, epsilon)
    t_safe = np.maximum(true_values, epsilon)
    q_errors = np.maximum(p_safe / t_safe, t_safe / p_safe)

    return {
        'predictions': predictions,
        'true_values': true_values,
        'predictions_log': predictions_log,
        'true_values_log': true_values_log,
        'mse': mse,
        'mae': metrics.calculate_mae(predictions, true_values),
        'rmse': np.sqrt(mse),
        'q_error': np.mean(q_errors),
        'median_mse': np.median((predictions - true_values)**2),
        'median_mae': np.median(np.abs(predictions - true_values)),
        'median_q_error': np.median(q_errors),
        'q_errors_array': q_errors
    }


class MetricLogger:
    def __init__(self, log_dir: Path, use_wandb: bool = False):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        
        # History storage (metrics only, no individual predictions)
        self.history = {
            'train_metrics': [],
            'test_metrics': [],
            'train_per_graph_metrics': [],
            'test_per_graph_metrics': [],
            'train_per_query_type_metrics': [],
            'test_per_query_type_metrics': [],
            'train_per_query_set_name_metrics': [],
            'test_per_query_set_name_metrics': [],
            'final_test_metrics': [],
            'final_test_per_graph_metrics': [],
            'final_test_per_query_type_metrics': [],
            'final_test_per_query_set_name_metrics': []
        }
        
        # Current epoch buffers
        self.current_epoch_data = {}
        self.current_epoch_query_type_data = {}
        self.current_epoch_query_set_name_data = {}
        self.current_stage = None
        self.current_epoch = None
        
        # Predictions storage - buffered in memory, written with pandas
        self.predictions = []

    def start_epoch(self, epoch: int = None, stage: str = None):
        """Resets internal buffers for a new epoch."""
        self.current_epoch_data = {}
        self.current_epoch_query_type_data = {}
        self.current_epoch_query_set_name_data = {}
        if epoch is not None:
            self.current_epoch = epoch
        if stage is not None:
            self.current_stage = stage

    def log_batch(self, predictions_log, true_values_log, loss, dataset_name: str, 
                  query_type: Literal["star", "path"] = None, query_set_name: str = None):
        """Store predictions in memory and accumulate batch data for metrics."""
        # Convert tensors to numpy
        if torch.is_tensor(predictions_log):
            predictions_log = predictions_log.detach().cpu().numpy()
        else:
            predictions_log = np.array(predictions_log)
            
        if torch.is_tensor(true_values_log):
            true_values_log = true_values_log.detach().cpu().numpy()
        else:
            true_values_log = np.array(true_values_log)
            
        if torch.is_tensor(loss):
            loss = loss.item()
        
        # Handle query_type if it's a list/array (batched data)
        if query_type is not None:
            if isinstance(query_type, (list, tuple, np.ndarray)):
                query_type = query_type[0] if len(query_type) > 0 else None
            elif torch.is_tensor(query_type):
                query_type = query_type[0].item() if query_type.numel() > 0 else None
        
        # Handle query_set_name if it's a list/array (batched data)
        if query_set_name is not None:
            if isinstance(query_set_name, (list, tuple, np.ndarray)):
                query_set_name = query_set_name[0] if len(query_set_name) > 0 else None
            elif torch.is_tensor(query_set_name):
                query_set_name = query_set_name[0].item() if query_set_name.numel() > 0 else None
        
        # Store predictions in memory
        predictions = np.expm1(predictions_log)
        true_values = np.expm1(true_values_log)
        for i, (pred_log, true_log, pred, true) in enumerate(zip(
            predictions_log.tolist(), true_values_log.tolist(), 
            predictions.tolist(), true_values.tolist()
        )):
            self.predictions.append({
                'stage': self.current_stage,
                'epoch': self.current_epoch,
                'dataset_name': dataset_name,
                'query_set_name': query_set_name if query_set_name else '',
                'query_type': query_type if query_type else '',
                'batch_idx': i,
                'prediction': pred,
                'prediction_log': pred_log,
                'true_value': true,
                'true_value_log': true_log
            })
        
        # Store aggregated data for metrics calculation (no individual predictions)
        if dataset_name not in self.current_epoch_data:
            self.current_epoch_data[dataset_name] = {
                'predictions_log': [],
                'true_values_log': [],
                'loss_sum': 0.0,
                'batch_count': 0
            }
            
        data = self.current_epoch_data[dataset_name]
        data['predictions_log'].extend(predictions_log.tolist())
        data['true_values_log'].extend(true_values_log.tolist())
        data['loss_sum'] += loss
        data['batch_count'] += 1
        
        # Store in per-query-type data (only if query_type is not None)
        if query_type is not None:
            key = (dataset_name, query_type)
            if key not in self.current_epoch_query_type_data:
                self.current_epoch_query_type_data[key] = {
                    'predictions_log': [],
                    'true_values_log': [],
                    'loss_sum': 0.0,
                    'batch_count': 0
                }
            
            qt_data = self.current_epoch_query_type_data[key]
            qt_data['predictions_log'].extend(predictions_log.tolist())
            qt_data['true_values_log'].extend(true_values_log.tolist())
            qt_data['loss_sum'] += loss
            qt_data['batch_count'] += 1
        
        # Store in per-query-set-name data (only if query_set_name is not None)
        if query_set_name is not None:
            key = (dataset_name, query_set_name)
            if key not in self.current_epoch_query_set_name_data:
                self.current_epoch_query_set_name_data[key] = {
                    'predictions_log': [],
                    'true_values_log': [],
                    'loss_sum': 0.0,
                    'batch_count': 0
                }
            
            qs_data = self.current_epoch_query_set_name_data[key]
            qs_data['predictions_log'].extend(predictions_log.tolist())
            qs_data['true_values_log'].extend(true_values_log.tolist())
            qs_data['loss_sum'] += loss
            qs_data['batch_count'] += 1
    
    # TODO:  Takes ages
    def end_epoch(self, epoch: int, stage: str = 'train', save: bool = False) -> Dict[str, Any]:
        """
        Processes accumulated data, calculates metrics, logs to wandb, and updates history.
        """
        per_graph_metrics = {}
        per_query_type_metrics = {}
        per_query_set_name_metrics = {}
        all_predictions_log = []
        all_true_values_log = []
        total_loss_sum = 0
        total_batches = 0
        
        # Process per-graph data
        for dataset_name, data in self.current_epoch_data.items():
            predictions_log = np.array(data['predictions_log'])
            true_values_log = np.array(data['true_values_log'])
            
            # Calculate metrics for this graph
            graph_metrics_results = calculate_all_metrics(predictions_log, true_values_log)
            avg_loss = data['loss_sum'] / data['batch_count']
            
            per_graph_metrics[dataset_name] = {
                'loss': avg_loss,
                **graph_metrics_results
            }
            
            # Accumulate for overall metrics
            all_predictions_log.extend(data['predictions_log'])
            all_true_values_log.extend(data['true_values_log'])
            total_loss_sum += data['loss_sum']
            total_batches += data['batch_count']
            
            # Store per-graph metrics in history
            self.history[f'{stage}_per_graph_metrics'].append({
                'epoch': epoch,
                'dataset_name': dataset_name,
                'loss': avg_loss,
                'mse': graph_metrics_results['mse'],
                'mae': graph_metrics_results['mae'],
                'rmse': graph_metrics_results['rmse'],
                'q_error': graph_metrics_results['q_error']
            })
        
        # Process per-query-type data
        for (dataset_name, query_type), data in self.current_epoch_query_type_data.items():
            predictions_log = np.array(data['predictions_log'])
            true_values_log = np.array(data['true_values_log'])
            
            # Calculate metrics for this dataset-query_type combination
            qt_metrics_results = calculate_all_metrics(predictions_log, true_values_log)
            avg_loss = data['loss_sum'] / data['batch_count']
            
            per_query_type_metrics[(dataset_name, query_type)] = {
                'loss': avg_loss,
                **qt_metrics_results
            }
            
            # Store per-query-type metrics in history
            self.history[f'{stage}_per_query_type_metrics'].append({
                'epoch': epoch,
                'dataset_name': dataset_name,
                'query_type': query_type,
                'loss': avg_loss,
                'mse': qt_metrics_results['mse'],
                'mae': qt_metrics_results['mae'],
                'rmse': qt_metrics_results['rmse'],
                'q_error': qt_metrics_results['q_error']
            })
        
        # Process per-query-set-name data
        for (dataset_name, query_set_name), data in self.current_epoch_query_set_name_data.items():
            predictions_log = np.array(data['predictions_log'])
            true_values_log = np.array(data['true_values_log'])
            
            # Calculate metrics for this dataset-query_set_name combination
            qs_metrics_results = calculate_all_metrics(predictions_log, true_values_log)
            avg_loss = data['loss_sum'] / data['batch_count']
            
            per_query_set_name_metrics[(dataset_name, query_set_name)] = {
                'loss': avg_loss,
                **qs_metrics_results
            }
            
            # Store per-query-set-name metrics in history
            self.history[f'{stage}_per_query_set_name_metrics'].append({
                'epoch': epoch,
                'dataset_name': dataset_name,
                'query_set_name': query_set_name,
                'loss': avg_loss,
                'mse': qs_metrics_results['mse'],
                'mae': qs_metrics_results['mae'],
                'rmse': qs_metrics_results['rmse'],
                'q_error': qs_metrics_results['q_error']
            })

        # Calculate overall metrics
        overall_metrics = calculate_all_metrics(all_predictions_log, all_true_values_log)
        overall_loss = total_loss_sum / total_batches if total_batches > 0 else 0
        overall_metrics['loss'] = overall_loss
        
        # Store overall metrics in history
        self.history[f'{stage}_metrics'].append({
            'epoch': epoch,
            'loss': overall_loss,
            'mse': overall_metrics['mse'],
            'mae': overall_metrics['mae'],
            'rmse': overall_metrics['rmse'],
            'q_error': overall_metrics['q_error']
        })
        
        # Log to WandB (only when saving to reduce overhead)
        if self.use_wandb and save:
            self._log_to_wandb(epoch, stage, overall_metrics, per_graph_metrics, per_query_type_metrics, per_query_set_name_metrics)

        # Save CSVs
        if save:
            self.save_csvs()
            
        return overall_metrics
    
    def _wandb_log_retry(self, data):
        for _ in range(200):
            try:
                wandb.log(data)
                return
            except Exception as e:
                logger.error(f"Error logging to wandb: {e}. Retrying...")
                time.sleep(1)

    def log_embedding_similarity(self, embeddings, name: str, epoch: int, sample_size: int = 2000):
        """
        Analyzes pairwise cosine similarity of a set of vectors and logs to WandB.
        
        Args:
            embeddings: Tensor of shape [N, D]
            name: String identifier for plots/logs
            epoch: Current epoch number
            sample_size: Max number of vectors to use
        """
        if not self.use_wandb:
            return

        if embeddings.size(0) < 2:
            logger.warning(f"[{name}] Not enough vectors to compute similarity.")
            return

        try:
            # Detach and move to computation device
            vecs = embeddings.detach()
            
            # Optional: Downsample if too many vectors
            if vecs.size(0) > sample_size:
                perm = torch.randperm(vecs.size(0))
                vecs = vecs[perm[:sample_size]]
            
            # 1. Normalize vectors to unit length (L2 norm)
            vecs_norm = torch.nn.functional.normalize(vecs, p=2, dim=1)
            
            # 2. Compute Pairwise Similarity Matrix (N x N)
            similarity_matrix = torch.mm(vecs_norm, vecs_norm.t())
            
            # 3. Extract Upper Triangle (off-diagonal)
            triu_indices = torch.triu_indices(similarity_matrix.size(0), similarity_matrix.size(1), offset=1)
            similarities = similarity_matrix[triu_indices[0], triu_indices[1]]
            
            # --- Metrics ---
            avg_sim = similarities.mean().item()
            std_sim = similarities.std().item()
            min_sim = similarities.min().item()
            max_sim = similarities.max().item()
            
            # Log scalar metrics
            wandb.log({
                f"{name}_avg_similarity": avg_sim,
                f"{name}_std_similarity": std_sim,
                f"{name}_min_similarity": min_sim,
                f"{name}_max_similarity": max_sim,
                "epoch": epoch
            })
            
            # --- Plotting ---
            # Move to CPU for plotting
            sim_values = similarities.cpu().numpy()
            
            # Histogram
            fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
            ax_hist.hist(sim_values, bins=50, range=(-1, 1), alpha=0.7, color='skyblue', edgecolor='black')
            ax_hist.axvline(avg_sim, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {avg_sim:.2f}')
            ax_hist.set_title(f'Pairwise Cosine Similarity Distribution: {name}')
            ax_hist.set_xlabel('Cosine Similarity (-1 to 1)')
            ax_hist.set_ylabel('Frequency')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)
            
            wandb.log({
                f"{name}_similarity_hist": wandb.Image(fig_hist),
                "epoch": epoch
            })
            plt.close(fig_hist)
            
            # PCA Projection
            vecs_np = vecs.cpu().numpy()
            pca = PCA(n_components=2)
            vecs_2d = pca.fit_transform(vecs_np)
            
            fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
            ax_pca.scatter(vecs_2d[:, 0], vecs_2d[:, 1], alpha=0.6, s=10, color='teal')
            ax_pca.set_title(f'PCA Projection (2D): {name}')
            ax_pca.set_xlabel('PC1')
            ax_pca.set_ylabel('PC2')
            ax_pca.grid(True, alpha=0.3)
            
            wandb.log({
                f"{name}_pca_projection": wandb.Image(fig_pca),
                "epoch": epoch
            })
            plt.close(fig_pca)

            # t-SNE Projection
            try:
                # Use a smaller perplexity if n_samples is small, typically between 5 and 50
                n_samples = vecs_np.shape[0]
                perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
                
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
                vecs_tsne = tsne.fit_transform(vecs_np)
                
                fig_tsne, ax_tsne = plt.subplots(figsize=(8, 6))
                ax_tsne.scatter(vecs_tsne[:, 0], vecs_tsne[:, 1], alpha=0.6, s=10, color='purple')
                ax_tsne.set_title(f't-SNE Projection (2D): {name}')
                ax_tsne.set_xlabel('Dim 1')
                ax_tsne.set_ylabel('Dim 2')
                ax_tsne.grid(True, alpha=0.3)
                
                wandb.log({
                    f"{name}_tsne_projection": wandb.Image(fig_tsne),
                    "epoch": epoch
                })
                plt.close(fig_tsne)
            except Exception as e:
                logger.error(f"Could not compute/plot t-SNE: {e}")
            
        except Exception as e:
            logger.error(f"Error logging embedding similarity for {name}: {e}")

    def _log_to_wandb(self, epoch, stage, overall_metrics, per_graph_metrics, per_query_type_metrics, per_query_set_name_metrics):
        # Log overall/general metrics
        log_data = {
            f"{stage}/overall/loss": overall_metrics['loss'],
            f"{stage}/overall/mse": overall_metrics['mse'],
            f"{stage}/overall/mae": overall_metrics['mae'],
            f"{stage}/overall/rmse": overall_metrics['rmse'],
            f"{stage}/overall/q_error": overall_metrics['q_error'],
            f"{stage}/overall/median_mse": overall_metrics['median_mse'],
            f"{stage}/overall/median_mae": overall_metrics['median_mae'],
            f"{stage}/overall/median_q_error": overall_metrics['median_q_error'],
            'epoch': epoch
        }
        self._wandb_log_retry(log_data)
        
        # Log overall q-error boxplot
        self._log_q_error_boxplot(
            overall_metrics['q_errors_array'],
            overall_metrics['true_values'],
            epoch, f"{stage}/overall"
        )

        # Log per-graph metrics
        for dataset_name, graph_metrics in per_graph_metrics.items():
            graph_log_data = {
                f"{stage}/per_graph/{dataset_name}/loss": graph_metrics['loss'],
                f"{stage}/per_graph/{dataset_name}/mse": graph_metrics['mse'],
                f"{stage}/per_graph/{dataset_name}/mae": graph_metrics['mae'],
                f"{stage}/per_graph/{dataset_name}/rmse": graph_metrics['rmse'],
                f"{stage}/per_graph/{dataset_name}/q_error": graph_metrics['q_error'],
                f"{stage}/per_graph/{dataset_name}/median_mse": graph_metrics['median_mse'],
                f"{stage}/per_graph/{dataset_name}/median_mae": graph_metrics['median_mae'],
                f"{stage}/per_graph/{dataset_name}/median_q_error": graph_metrics['median_q_error'],
                'epoch': epoch
            }
            self._wandb_log_retry(graph_log_data)
            
            # Log per-graph q-error boxplot
            self._log_q_error_boxplot(
                graph_metrics['q_errors_array'],
                graph_metrics['true_values'],
                epoch, f"{stage}/per_graph/{dataset_name}"
            )

            # Log predictions scatter plot (sampled)
            self._log_predictions_scatter(
                graph_metrics['predictions_log'], 
                graph_metrics['true_values_log'], 
                epoch, stage, f"per_graph/{dataset_name}"
            )
        
        # Log per-query-type metrics
        for (dataset_name, query_type), qt_metrics in per_query_type_metrics.items():
            qt_log_data = {
                f"{stage}/per_query_type/{dataset_name}_{query_type}/loss": qt_metrics['loss'],
                f"{stage}/per_query_type/{dataset_name}_{query_type}/mse": qt_metrics['mse'],
                f"{stage}/per_query_type/{dataset_name}_{query_type}/mae": qt_metrics['mae'],
                f"{stage}/per_query_type/{dataset_name}_{query_type}/rmse": qt_metrics['rmse'],
                f"{stage}/per_query_type/{dataset_name}_{query_type}/q_error": qt_metrics['q_error'],
                f"{stage}/per_query_type/{dataset_name}_{query_type}/median_mse": qt_metrics['median_mse'],
                f"{stage}/per_query_type/{dataset_name}_{query_type}/median_mae": qt_metrics['median_mae'],
                f"{stage}/per_query_type/{dataset_name}_{query_type}/median_q_error": qt_metrics['median_q_error'],
                'epoch': epoch
            }
            self._wandb_log_retry(qt_log_data)
            
            # Log per-query-type q-error boxplot
            self._log_q_error_boxplot(
                qt_metrics['q_errors_array'],
                qt_metrics['true_values'],
                epoch, f"{stage}/per_query_type/{dataset_name}_{query_type}"
            )
            
            # Log per-query-type predictions scatter plot
            self._log_predictions_scatter(
                qt_metrics['predictions_log'],
                qt_metrics['true_values_log'],
                epoch, stage, f"per_query_type/{dataset_name}_{query_type}"
            )
        
        # Log per-query-set-name metrics
        for (dataset_name, query_set_name), qs_metrics in per_query_set_name_metrics.items():
            qs_log_data = {
                f"{stage}/per_query_set_name/{dataset_name}_{query_set_name}/loss": qs_metrics['loss'],
                f"{stage}/per_query_set_name/{dataset_name}_{query_set_name}/mse": qs_metrics['mse'],
                f"{stage}/per_query_set_name/{dataset_name}_{query_set_name}/mae": qs_metrics['mae'],
                f"{stage}/per_query_set_name/{dataset_name}_{query_set_name}/rmse": qs_metrics['rmse'],
                f"{stage}/per_query_set_name/{dataset_name}_{query_set_name}/q_error": qs_metrics['q_error'],
                f"{stage}/per_query_set_name/{dataset_name}_{query_set_name}/median_mse": qs_metrics['median_mse'],
                f"{stage}/per_query_set_name/{dataset_name}_{query_set_name}/median_mae": qs_metrics['median_mae'],
                f"{stage}/per_query_set_name/{dataset_name}_{query_set_name}/median_q_error": qs_metrics['median_q_error'],
                'epoch': epoch
            }
            self._wandb_log_retry(qs_log_data)
            
            # Log per-query-set-name q-error boxplot
            self._log_q_error_boxplot(
                qs_metrics['q_errors_array'],
                qs_metrics['true_values'],
                epoch, f"{stage}/per_query_set_name/{dataset_name}_{query_set_name}"
            )
            
            # Log per-query-set-name predictions scatter plot
            self._log_predictions_scatter(
                qs_metrics['predictions_log'],
                qs_metrics['true_values_log'],
                epoch, stage, f"per_query_set_name/{dataset_name}_{query_set_name}"
            )
            
        # Log overall predictions scatter plot
        self._log_predictions_scatter(
            overall_metrics['predictions_log'],
            overall_metrics['true_values_log'],
            epoch, stage, "overall"
        )

    def _log_q_error_boxplot(self, q_errors, true_values, epoch, title_prefix):
        """Helper to plot Q-Error boxplots by cardinality range."""
        bins = [
            ('< 10', (true_values < 10)),
            ('< 100', (true_values >= 10) & (true_values < 100)),
            ('< 1000', (true_values >= 100) & (true_values < 1000)),
            ('< 1M', (true_values >= 1000) & (true_values < 1000000)),
            ('Other', (true_values >= 1000000))
        ]
        
        data_to_plot = []
        labels = []
        
        for label, mask in bins:
            if np.any(mask):
                data_to_plot.append(q_errors[mask])
                labels.append(f"{label}\n(n={np.sum(mask)})")
        
        if not data_to_plot:
            return

        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(data_to_plot, labels=labels)
            ax.set_title(f"Q-Error Distribution by Cardinality - {title_prefix}")
            ax.set_ylabel("Q-Error")
            ax.set_xlabel("True Cardinality Range")
            ax.set_yscale('log')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            self._wandb_log_retry({
                f"{title_prefix}_q_error_boxplot": wandb.Image(fig),
                "epoch": epoch
            })
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error plotting boxplot: {e}")

    def _log_predictions_scatter(self, predictions, true_values, epoch, stage, dataset_name, max_points=1000):
        try:
            predictions = np.array(predictions)
            true_values = np.array(true_values)
            
            valid_mask = np.isfinite(predictions) & np.isfinite(true_values)
            predictions_clean = predictions[valid_mask]
            true_values_clean = true_values[valid_mask]
            
            if len(predictions_clean) == 0:
                return
            
            if len(predictions_clean) > max_points:
                indices = np.random.choice(len(predictions_clean), max_points, replace=False)
                predictions_sampled = predictions_clean[indices]
                true_values_sampled = true_values_clean[indices]
            else:
                predictions_sampled = predictions_clean
                true_values_sampled = true_values_clean
            
            data = [[float(pred), float(true)] for pred, true in zip(predictions_sampled, true_values_sampled)]
            table = wandb.Table(data=data, columns=["prediction", "true_value"])
            self._wandb_log_retry({
                f"{stage}_predictions_{dataset_name}_epoch_{epoch}": table,
                "epoch": epoch
            })
        except Exception as e:
            logger.error(f"Error logging predictions to wandb: {e}")

    def save_csvs(self):
        """Save all accumulated metrics and predictions to CSV files."""
        files_map = {
            'train_metrics': TRAIN_METRICS_FILE_NAME,
            'test_metrics': TEST_METRICS_FILE_NAME,
            'final_test_metrics': 'final_test_metrics.csv',
            'train_per_graph_metrics': TRAIN_METRICS_PER_GRAPH_FILE_NAME,
            'test_per_graph_metrics': TEST_METRICS_PER_GRAPH_FILE_NAME,
            'final_test_per_graph_metrics': 'final_test_metrics_per_graph.csv',
            'train_per_query_type_metrics': TRAIN_METRICS_PER_QUERY_TYPE_FILE_NAME,
            'test_per_query_type_metrics': TEST_METRICS_PER_QUERY_TYPE_FILE_NAME,
            'final_test_per_query_type_metrics': 'final_test_metrics_per_query_type.csv',
            'train_per_query_set_name_metrics': TRAIN_METRICS_PER_QUERY_SET_NAME_FILE_NAME,
            'test_per_query_set_name_metrics': TEST_METRICS_PER_QUERY_SET_NAME_FILE_NAME,
            'final_test_per_query_set_name_metrics': 'final_test_metrics_per_query_set_name.csv'
        }
        
        for key, filename in files_map.items():
            data = self.history[key]
            if data:
                df = pd.DataFrame(data)
                df.to_csv(self.log_dir / filename, index=False)
        
        # Save predictions with pandas
        if self.predictions:
            df = pd.DataFrame(self.predictions)
            df.to_csv(self.log_dir / PREDICTIONS_FILE_NAME, index=False)

    def get_best_test_loss(self) -> float:
        """Returns the best test loss recorded so far."""
        if not self.history['test_metrics']:
            return float('inf')
        return min(m['loss'] for m in self.history['test_metrics'])
