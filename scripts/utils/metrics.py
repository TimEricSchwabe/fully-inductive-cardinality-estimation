import numpy as np
import torch

def calculate_mae(predictions, true_values):
    return np.mean(np.abs(np.array(predictions) - np.array(true_values)))

def calculate_mse(predictions, true_values):
    return np.mean((np.array(predictions) - np.array(true_values))**2)

def calculate_q_error(predictions, true_values, epsilon=1e-7):
    pred = np.maximum(np.array(predictions), epsilon)
    true = np.maximum(np.array(true_values), epsilon)
    return np.mean(np.maximum(pred / true, true / pred))

def torch_q_error_loss(predictions, true_values, epsilon=1e-7):
    predictions = torch.maximum(predictions, torch.tensor(epsilon))
    true_values = torch.maximum(true_values, torch.tensor(epsilon))
    return torch.mean(torch.maximum(predictions / true_values, true_values / predictions))

