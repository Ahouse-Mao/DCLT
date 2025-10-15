import numpy as np
from typing import Tuple

_EPS = 1e-8


def calc_forecast_metrics(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    """Compute MAE/MSE/RMSE/MAPE/MSPE/RSE/CORR for flattened predictions."""
    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    true_arr = np.asarray(true, dtype=np.float64).reshape(-1)

    diff = pred_arr - true_arr

    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    denom_abs = np.abs(true_arr) + _EPS
    mape = float(np.mean(np.abs(diff) / denom_abs))
    mspe = float(np.mean((diff / (true_arr + _EPS)) ** 2))

    rse_denom = np.sqrt(np.sum((true_arr - true_arr.mean()) ** 2)) + _EPS
    rse = float(np.sqrt(np.sum(diff ** 2)) / rse_denom)

    true_centered = true_arr - true_arr.mean()
    pred_centered = pred_arr - pred_arr.mean()
    corr_denom = np.sqrt(np.sum(true_centered ** 2) * np.sum(pred_centered ** 2)) + _EPS
    corr = float(np.sum(true_centered * pred_centered) / corr_denom)

    return mae, mse, rmse, mape, mspe, rse, corr
