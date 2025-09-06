import numpy as np
import torch

def seed_everything(seed: int = 42):
    """
    Set the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def print_cfg(cfg, logger):
    """
    Print the configuration.
    """
    for key, value in cfg.items():
        logger.info(f"{key}: {value}")
