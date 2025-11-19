"""
Helper functions for logging metrics, saving CSVs, and managing training data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd
import torch
import wandb

import utils.metrics as metrics


logger = logging.getLogger(__name__)


# CSV file name constants
TRAIN_METRICS_FILE_NAME = "train_training_metrics.csv"
TEST_METRICS_FILE_NAME = "test_training_metrics.csv"
TRAIN_METRICS_PER_GRAPH_FILE_NAME = "train_metrics_per_graph.csv"
TEST_METRICS_PER_GRAPH_FILE_NAME = "test_metrics_per_graph.csv"
TRAIN_VALUES_PER_GRAPH_FILE_NAME = "train_values_per_graph.csv"
TEST_VALUES_PER_GRAPH_FILE_NAME = "test_values_per_graph.csv"


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
    predictions = np.exp(predictions_log)
    true_values = np.exp(true_values_log)
    mse = metrics.calculate_mse(predictions, true_values)
    
    return {
        'predictions': predictions,
        'true_values': true_values,
        'predictions_log': predictions_log,
        'true_values_log': true_values_log,
        'mse': mse,
        'mae': metrics.calculate_mae(predictions, true_values),
        'rmse': np.sqrt(mse),
        'q_error': metrics.calculate_q_error(predictions, true_values)
    }


class MetricLogger:
    def __init__(self, log_dir: Path, use_wandb: bool = False):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        
        # History storage
        self.history = {
            'train_metrics': [],
            'test_metrics': [],
            'train_per_graph_metrics': [],
            'test_per_graph_metrics': [],
            'train_predictions': [],
            'test_predictions': [],

            'final_test_metrics': [],
            'final_test_per_graph_metrics': [],
            'final_test_predictions': []
        }
        
        # Current epoch buffers
        self.current_epoch_data = {}

    def start_epoch(self):
        """Resets internal buffers for a new epoch."""
        self.current_epoch_data = {}

    def log_batch(self, predictions_log, true_values_log, loss, dataset_name: str):
        """Accumulates batch data."""
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

        # Convert to list for storage
        predictions_log = predictions_log.tolist()
        true_values_log = true_values_log.tolist()
            
        if dataset_name not in self.current_epoch_data:
            self.current_epoch_data[dataset_name] = {
                'predictions_log': [],
                'true_values_log': [],
                'loss_sum': 0.0,
                'batch_count': 0
            }
            
        data = self.current_epoch_data[dataset_name]
        data['predictions_log'].extend(predictions_log)
        data['true_values_log'].extend(true_values_log)
        data['loss_sum'] += loss
        data['batch_count'] += 1

    def end_epoch(self, epoch: int, stage: str = 'train', save: bool = False) -> Dict[str, Any]:
        """
        Processes accumulated data, calculates metrics, logs to wandb, and updates history.
        """
        per_graph_metrics = {}
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
            
            # Store per-graph predictions in history
            for i, (pred_log, true_log, pred, true) in enumerate(zip(
                graph_metrics_results['predictions_log'],
                graph_metrics_results['true_values_log'],
                graph_metrics_results['predictions'],
                graph_metrics_results['true_values']
            )):
                self.history[f'{stage}_predictions'].append({
                    'epoch': epoch,
                    'dataset_name': dataset_name,
                    'batch_idx': i,
                    'prediction_log': pred_log,
                    'true_value_log': true_log,
                    'prediction': pred,
                    'true_value': true
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
        
        # Log to WandB
        if self.use_wandb:
            self._log_to_wandb(epoch, stage, overall_metrics, per_graph_metrics)

        # Save CSVs
        if save:
            self.save_csvs()
            
        return overall_metrics

    def _log_to_wandb(self, epoch, stage, overall_metrics, per_graph_metrics):
        # Log overall metrics
        log_data = {
            f"{stage}_loss": overall_metrics['loss'],
            f"{stage}_mse": overall_metrics['mse'],
            f"{stage}_mae": overall_metrics['mae'],
            f"{stage}_rmse": overall_metrics['rmse'],
            f"{stage}_q_error": overall_metrics['q_error'],
            'epoch': epoch
        }
        wandb.log(log_data)
        
        # Log per-graph metrics
        for dataset_name, graph_metrics in per_graph_metrics.items():
            graph_log_data = {
                f"{stage}_loss_{dataset_name}": graph_metrics['loss'],
                f"{stage}_mse_{dataset_name}": graph_metrics['mse'],
                f"{stage}_mae_{dataset_name}": graph_metrics['mae'],
                f"{stage}_rmse_{dataset_name}": graph_metrics['rmse'],
                f"{stage}_q_error_{dataset_name}": graph_metrics['q_error'],
                'epoch': epoch
            }
            wandb.log(graph_log_data)
            
            # Log predictions scatter plot (sampled)
            self._log_predictions_scatter(
                graph_metrics['predictions_log'], 
                graph_metrics['true_values_log'], 
                epoch, stage, dataset_name
            )
            
        # Log overall predictions scatter plot
        self._log_predictions_scatter(
            overall_metrics['predictions_log'],
            overall_metrics['true_values_log'],
            epoch, stage, "overall"
        )

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
            
            wandb.log({
                f"{stage}_predictions_{dataset_name}_epoch_{epoch}": table,
                "epoch": epoch
            })
        except Exception as e:
            logger.error(f"Error logging predictions to wandb: {e}")

    def save_csvs(self):
        """Save all accumulated history to CSV files."""
        files_map = {
            'train_metrics': TRAIN_METRICS_FILE_NAME,
            'test_metrics': TEST_METRICS_FILE_NAME,
            'final_test_metrics': 'final_test_metrics.csv',
            'train_per_graph_metrics': TRAIN_METRICS_PER_GRAPH_FILE_NAME,
            'test_per_graph_metrics': TEST_METRICS_PER_GRAPH_FILE_NAME,
            'final_test_per_graph_metrics': 'final_test_metrics_per_graph.csv',
            'train_predictions': TRAIN_VALUES_PER_GRAPH_FILE_NAME,
            'test_predictions': TEST_VALUES_PER_GRAPH_FILE_NAME,
            'final_test_predictions': 'final_test_predictions_per_graph.csv'
        }
        
        for key, filename in files_map.items():
            data = self.history[key]
            if data:
                df = pd.DataFrame(data)
                df.to_csv(self.log_dir / filename, index=False)

    def get_best_test_loss(self) -> float:
        """Returns the best test loss recorded so far."""
        if not self.history['test_metrics']:
            return float('inf')
        return min(m['loss'] for m in self.history['test_metrics'])
