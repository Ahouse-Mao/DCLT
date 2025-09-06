# DCLT Project

Deep Contrastive Learning for Time series (DCLT) project with PatchTST implementation.

## Project Structure

```
DCLT/
├── .gitignore                               # Git ignore rules for ML projects
├── DCLT_main_pretrain.py                    # Main DCLT pretraining script
├── DCLT_patchtst_pretrained_cl.py          # DCLT with pretrained PatchTST
├── PatchTST_supervised/                     # Original PatchTST implementation
│   ├── conf/                               # Modern YAML configuration system
│   │   ├── default_config.yaml            # Default parameters
│   │   ├── etth1_config.yaml              # ETTh1 dataset config
│   │   └── etth2_config.yaml              # ETTh2 dataset config
│   ├── run_main.py                         # Modern training script with YAML support
│   ├── run_longExp.py                      # Original training script
│   ├── README_modern_config.md             # Configuration system documentation
│   └── ...                                 # Other PatchTST files
├── cl_conf/                                 # Contrastive learning configurations
├── cl_models/                               # Contrastive learning models
├── data_provider/                           # Data loading utilities
├── dataset/                                 # Time series datasets
├── layers/                                  # Neural network layers
├── loss/                                    # Loss functions
└── utils/                                   # Utility functions
```

## Key Features

### 1. DCLT (Deep Contrastive Learning for Time series)
- Contrastive learning approach for time series representation learning
- Pre-training and fine-tuning pipeline
- Support for various time series datasets

### 2. Modern PatchTST Configuration System
- YAML-based configuration management
- Hierarchical configuration with inheritance
- Command-line parameter overrides
- Support for multiple experiments

### 3. Comprehensive Dataset Support
- ETT (Electricity Transformer Temperature) datasets
- Electricity consumption data
- Weather data
- Traffic data
- Exchange rate data
- National illness data

## Quick Start

### Traditional PatchTST Training
```bash
cd PatchTST_supervised
# Old way - using shell scripts
bash scripts/PatchTST/etth1.sh

# New way - using YAML configuration
python run_main.py --config conf/etth1_config.yaml
```

### DCLT Training
```bash
# DCLT pretraining
python DCLT_main_pretrain.py

# DCLT with pretrained PatchTST
python DCLT_patchtst_pretrained_cl.py
```

## Configuration System

The modern configuration system supports:

- **Default configurations**: Base parameters in `PatchTST_supervised/conf/default_config.yaml`
- **Dataset-specific configs**: Override parameters for specific datasets
- **Command-line overrides**: Quick parameter modifications without editing files
- **Multi-experiment support**: Run multiple prediction lengths in one command

Example usage:
```bash
# Basic usage
python run_main.py --config conf/etth1_config.yaml

# With overrides
python run_main.py --config conf/etth1_config.yaml --override learning_rate=0.001 batch_size=64

# Dry run to validate config
python run_main.py --config conf/etth1_config.yaml --dry_run
```

## Ignored Files

The `.gitignore` file is configured to ignore:
- Python cache files (`__pycache__/`, `*.pyc`)
- Model checkpoints (`*.pt`, `*.pth`, `*.ckpt`)
- Training logs and outputs
- Large result files
- Temporary and OS-specific files

But keeps:
- Source code
- Configuration files
- Essential datasets
- Documentation

## Dependencies

See `PatchTST_supervised/requirements.txt` for Python dependencies.

## Citation

If you use this code, please cite the relevant papers:
- PatchTST paper
- DCLT paper (if applicable)

## License

Please check individual components for their respective licenses.
