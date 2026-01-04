from copy import deepcopy
from typing import Union
from pathlib import Path

import wandb
from utils.models import *
import torch
import logging
from utils.data_helper import *
from utils.config import Config
import time
from utils.helper import wandb_log_artifact_with_retry

logger = logging.getLogger(__name__)

def save_checkpoint(state: TrainingState, checkpoint_path: Union[str, Path]):
    def to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_cpu(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(to_cpu(v) for v in obj)
        else:
            return obj

    if not isinstance(checkpoint_path, Path):
        checkpoint_path = Path(checkpoint_path)
     
    # Ensure the directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle file name conflicts by adding a counter if file already exists
    original_path = checkpoint_path
    counter = 0
    while checkpoint_path.exists():
        stem = original_path.stem
        suffix = original_path.suffix
        checkpoint_path = original_path.parent / f"{stem}_{counter}{suffix}"
        counter += 1

    model_state_dict = state.model_components.model.state_dict()
    optimizer_state_dict = state.model_components.optimizer.state_dict()
    
    training_state = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'config': state.training_context.config.model_dump(),
    }
    
    if state.model_components.scheduler is not None:
        training_state['scheduler_state_dict'] = state.model_components.scheduler.state_dict()

    torch.save(training_state, checkpoint_path)
    epoch = state.training_context.current_epoch

    # Log checkpoint as wandb artifact if enabled
    if getattr(state.training_context, 'use_wandb', False):
        try:
            artifact = wandb.Artifact(
                name=f"checkpoint_epoch_{epoch}",
                type="model",
                description=f"Checkpoint at epoch {epoch}"
            )
            artifact.add_file(str(checkpoint_path))
            
            wandb_log_artifact_with_retry(artifact)
                    
        except Exception as e:
            logger.error(f"Failed to prepare/log checkpoint artifact: {e}")


def load_checkpoint(checkpoint_path: Union[str, Path]) -> TrainingState:
    if not isinstance(checkpoint_path, Path):
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    training_state = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))

    return training_state