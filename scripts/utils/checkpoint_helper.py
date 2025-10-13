from typing import Union
from pathlib import Path

import wandb
from utils.models import *
import torch
import logging
from utils.data_helper import *
from utils.helper import set_random_seeds
import copy
logger = logging.getLogger(__name__)


def save_checkpoint(state: TrainingState, checkpoint_path: Union[str, Path]):
    if not isinstance(checkpoint_path, Path):
        checkpoint_path = Path(checkpoint_path)
     
    # Ensure the directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    copy_state = copy.deepcopy(state)
    wandb.unwatch(copy_state.model_components.model)

    # Handle file name conflicts by adding a counter if file already exists
    original_path = checkpoint_path
    counter = 0
    while checkpoint_path.exists():
        stem = original_path.stem
        suffix = original_path.suffix
        checkpoint_path = original_path.parent / f"{stem}_{counter}{suffix}"
        counter += 1

    copy_state.data_components = None

    torch.save(copy_state, checkpoint_path)
    epoch = state.training_context.current_epoch
    logger.info(f"Saved checkpoint for epoch {epoch} at {checkpoint_path}")


def load_checkpoint(checkpoint_path: Union[str, Path]):
    if not isinstance(checkpoint_path, Path):
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, weights_only=False)

    set_random_seeds(state.training_context.config["seed"])

    data = load_dataset(state.training_context)
    (entity_occurrences, relation_occurrences, entity_mapping, relation_mapping) = (
        load_occurrences(state.training_context)
    )
    
    queries = load_queries(
        state.training_context,
        entity_occurrences,
        relation_occurrences,
        entity_mapping,
        relation_mapping,
    )

    train_queries, test_queries = train_test_split(state.training_context, queries)
    train_loader = DataLoader(
        dataset=train_queries, batch_size=state.training_context.config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_queries,
        batch_size=1,  # Use batch_size=1 for evaluation
        shuffle=False,
    )

    # Create data components
    data_components = DataComponents(
        data=data,
        train_loader=train_loader,
        test_loader=test_loader,
        train_queries=train_queries,
        test_queries=test_queries,
    )
    state.data_components = data_components

    
    

    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return state