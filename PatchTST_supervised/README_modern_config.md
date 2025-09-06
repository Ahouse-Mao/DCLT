# Modern PatchTST Configuration System

## Overview
This directory contains a modernized version of the PatchTST training system that uses YAML configuration files instead of command-line arguments for better experiment management and reproducibility.

## Features
- **YAML Configuration**: Easy-to-read and version-controllable configuration files
- **Configuration Inheritance**: Default settings with experiment-specific overrides
- **Command-line Overrides**: Quickly modify parameters without editing files
- **Multi-experiment Support**: Run multiple prediction lengths in a single command
- **Dry Run Mode**: Validate configurations before training

## File Structure
```
conf/
├── default_config.yaml      # Default parameters for all experiments
├── etth1_config.yaml        # ETTh1 dataset specific configuration
├── etth2_config.yaml        # ETTh2 dataset specific configuration
└── ...                      # Add more dataset configurations as needed

run_main.py                  # Modern training script
run_longExp.py              # Original training script (unchanged)
```

## Usage

### Basic Usage
```bash
# Run ETTh1 experiment with all prediction lengths
python run_main.py --config conf/etth1_config.yaml

# Run ETTh2 experiment
python run_main.py --config conf/etth2_config.yaml
```

### Advanced Usage
```bash
# Run with command-line overrides
python run_main.py --config conf/etth1_config.yaml --override learning_rate=0.001 batch_size=64

# Run only specific prediction length
python run_main.py --config conf/etth1_config.yaml --pred_len 96

# Dry run to validate configuration
python run_main.py --config conf/etth1_config.yaml --dry_run

# Override nested configuration values
python run_main.py --config conf/etth1_config.yaml --override experiment.name=my_experiment train_epochs=50
```

## Configuration Files

### Default Configuration (`conf/default_config.yaml`)
Contains all default parameters that apply to most experiments. Other configuration files inherit from this.

### Experiment Configuration (e.g., `conf/etth1_config.yaml`)
- Inherits from default configuration
- Overrides dataset-specific parameters
- Defines experiment settings like prediction lengths
- Can specify logging directories

### Configuration Structure
```yaml
# Basic model settings
model: "PatchTST"
data: "ETTh1" 
data_path: "ETTh1.csv"

# Model architecture
d_model: 16
n_heads: 4
e_layers: 3

# Training settings
learning_rate: 0.0001
batch_size: 128
train_epochs: 100

# Experiment definition
experiment:
  name: "ETTh1_long_forecasting"
  pred_lens: [96, 192, 336, 720]  # Multiple prediction lengths
  seq_len: 336
```

## Creating New Experiments

1. Copy an existing configuration file (e.g., `etth1_config.yaml`)
2. Modify dataset and model parameters as needed
3. Update the experiment settings
4. Run with: `python run_main.py --config conf/your_config.yaml`

## Migration from Original System

The new system is fully compatible with the original `run_longExp.py`. Both can coexist:

- **Old way**: `python run_longExp.py --model PatchTST --data ETTh1 ...`
- **New way**: `python run_main.py --config conf/etth1_config.yaml`

All parameters from the original script are supported in the YAML configuration files.

## Benefits

1. **Reproducibility**: Configuration files can be version controlled
2. **Readability**: YAML is more readable than long command lines
3. **Reusability**: Share and reuse configurations easily
4. **Organization**: Group related experiments together
5. **Validation**: Dry run mode helps catch configuration errors early
6. **Flexibility**: Command-line overrides for quick parameter sweeps
