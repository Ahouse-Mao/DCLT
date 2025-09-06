#!/usr/bin/env python3
"""
Modern PatchTST Training Script with YAML Configuration Support
Based on the original run_longExp.py but with improved configuration management

Usage:
    python run_main.py --config conf/etth1_config.yaml
    python run_main.py --config conf/etth1_config.yaml --override learning_rate=0.001 batch_size=64
"""

import argparse
import os
import sys
import torch
import yaml
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from exp.exp_main import Exp_Main


class ConfigLoader:
    """Configuration loader that supports inheritance and overrides"""
    
    def __init__(self, config_dir: str = "conf"):
        self.config_dir = Path(config_dir)
        self.default_config_path = self.config_dir / "default_config.yaml"
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file with default inheritance"""
        # Load default configuration first
        config = self._load_yaml(self.default_config_path)
        
        # Load and merge experiment-specific configuration
        if config_path:
            exp_config = self._load_yaml(config_path)
            config = self._merge_configs(config, exp_config)
        
        return config
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            if file_path == self.default_config_path:
                raise FileNotFoundError(f"Default config file not found: {file_path}")
            return {}
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {file_path}: {e}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def apply_overrides(self, config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
        """Apply command line overrides to configuration"""
        for override in overrides:
            if '=' not in override:
                raise ValueError(f"Invalid override format: {override}. Expected key=value")
            
            key, value = override.split('=', 1)
            # Try to convert value to appropriate type
            try:
                # Try int first
                if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    value = int(value)
                # Try float
                elif '.' in value and value.replace('.', '').replace('-', '').isdigit():
                    value = float(value)
                # Try boolean
                elif value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                # Keep as string otherwise
            except ValueError:
                pass  # Keep as string
            
            # Set nested keys (e.g., experiment.name)
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        
        return config


class Args:
    """Configuration container that mimics argparse.Namespace"""
    
    def __init__(self, config: Dict[str, Any]):
        # Set all configuration values as attributes
        for key, value in config.items():
            if key != 'experiment':  # Skip experiment section
                setattr(self, key, value)
        
        # Handle special cases and validations
        self._post_process()
    
    def _post_process(self):
        """Post-process configuration values"""
        # GPU settings
        self.use_gpu = self.use_gpu and torch.cuda.is_available()
        
        if self.use_gpu and self.use_multi_gpu:
            self.devices = self.devices.replace(' ', '')
            device_ids = self.devices.split(',')
            self.device_ids = [int(id_) for id_ in device_ids]
            self.gpu = self.device_ids[0]
        
        # Ensure required directories exist
        os.makedirs(self.checkpoints, exist_ok=True)
        
        # Convert boolean-like integers
        bool_fields = ['revin', 'affine', 'subtract_last', 'decomposition', 'individual']
        for field in bool_fields:
            if hasattr(self, field):
                setattr(self, field, bool(getattr(self, field)))


def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_experiment_setting(args: Args, pred_len: int, iteration: int = 0) -> str:
    """Create experiment setting string for logging and checkpointing"""
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        f"{args.model_id}_{args.seq_len}_{pred_len}",
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        iteration
    )
    return setting


def run_experiment(args: Args, pred_len: int, iteration: int = 0, log_dir: str = None):
    """Run a single experiment with given prediction length"""
    # Update args with current prediction length
    args.pred_len = pred_len
    args.model_id = f"{args.data}_{args.seq_len}_{pred_len}"
    
    # Create experiment setting
    setting = create_experiment_setting(args, pred_len, iteration)
    
    # Initialize experiment
    exp = Exp_Main(args)
    
    print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    if args.is_training:
        exp.train(setting)
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting)
        
        if args.do_predict:
            print(f'>>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.predict(setting, True)
    else:
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
    
    # Clean up GPU memory
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Modern PatchTST Training with YAML Configuration')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to YAML configuration file')
    parser.add_argument('--override', nargs='*', default=[], 
                       help='Override configuration values (e.g., learning_rate=0.001 batch_size=64)')
    parser.add_argument('--pred_len', type=int, default=None,
                       help='Single prediction length to run (overrides config)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print configuration and exit without training')
    
    cmd_args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config(cmd_args.config)
    
    # Apply command line overrides
    if cmd_args.override:
        config = config_loader.apply_overrides(config, cmd_args.override)
    
    # Create args object
    args = Args(config)
    
    # Set random seed
    set_random_seed(args.random_seed)
    
    # Print configuration
    print('Configuration:')
    print(yaml.dump(config, default_flow_style=False, indent=2))
    
    if cmd_args.dry_run:
        print("Dry run mode - exiting without training")
        return
    
    # Determine prediction lengths to run
    pred_lens = [cmd_args.pred_len] if cmd_args.pred_len else config.get('experiment', {}).get('pred_lens', [args.pred_len])
    
    # Create log directory if specified
    log_dir = config.get('log_dir')
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Run experiments
    for pred_len in pred_lens:
        print(f"\n{'='*80}")
        print(f"Running experiment with prediction length: {pred_len}")
        print(f"{'='*80}")
        
        for iteration in range(args.itr):
            if args.itr > 1:
                print(f"\nIteration {iteration + 1}/{args.itr}")
            
            try:
                run_experiment(args, pred_len, iteration, log_dir)
            except Exception as e:
                print(f"Error in experiment pred_len={pred_len}, iteration={iteration}: {e}")
                if args.itr == 1:  # If only one iteration, re-raise the error
                    raise
                continue
    
    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
