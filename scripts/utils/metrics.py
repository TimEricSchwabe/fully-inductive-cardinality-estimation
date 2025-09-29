import numpy as np

def calculate_mae(predictions, true_values):
    return np.mean(np.abs(np.array(predictions) - np.array(true_values)))

def calculate_mse(predictions, true_values):
    return np.mean((np.array(predictions) - np.array(true_values))**2)

def calculate_q_error(predictions, true_values):
    q_errors = []
    eps = 1e-7
    for pred, true in zip(predictions, true_values):
        pred = max(pred, eps)
        true = max(true, eps)
        q_errors.append(max(pred / (true), true / (pred)))
    return np.mean(q_errors)


