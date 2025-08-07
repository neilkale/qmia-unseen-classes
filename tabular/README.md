# Tabular QMIA with Class Dropout

This directory contains scripts for running QMIA (Quantified Membership Inference Attack) experiments on tabular datasets with class dropout functionality.

## Overview

The pipeline consists of:
1. **Base Model Training**: Train an MLP classifier on the full dataset
2. **QMIA Training**: Train a quantile regression model excluding certain classes (dropped classes)
3. **Evaluation**: Measure attack performance on both seen and unseen classes

## Scripts

- `tabular_base_training.py`: Train base MLP model
- `tabular_qmia_training.py`: Train QMIA model with class dropout
- `tabular_runner.py`: Orchestrate the full pipeline
- `tabular_evaluation.py`: Analyze and visualize results

## Datasets

### Purchase-100
- Location: `data/dataset_purchase`
- Format: CSV with first column as class label, remaining columns as binary features
- Classes: 100 classes (0-99)

### Texas-100  
- Location: `data/texas/100/`
- Format: Binary files (`feats` and `labels`)
- Classes: 100 classes

## Usage

### Quick Start

Run the full pipeline for Texas dataset:
```bash
python tabular/tabular_runner.py --dataset texas
```

Run the full pipeline for Purchase dataset:
```bash
python tabular/tabular_runner.py --dataset purchase
```

### Custom Class Scenarios

Run with specific dropped classes:
```bash
python tabular/tabular_runner.py --dataset texas --custom_dropped_classes 90 91 92 93 94 95 96 97 98 99
```

Run with custom dropped ratios:
```bash
python tabular/tabular_runner.py --dataset texas --dropped_ratios 0.1 0.3 0.7 0.9
```

### Individual Scripts

#### 1. Base Model Training
```bash
python tabular/tabular_base_training.py --dataset texas --epochs 50
```

#### 2. QMIA Training
```bash
python tabular/tabular_qmia_training.py --dataset texas --dropped_classes 90 91 92 93 94 95 96 97 98 99
```

#### 3. Results Analysis
```bash
python tabular/tabular_evaluation.py --dataset texas --save_plots --save_table
```

## Key Features

1. **Class Dropout**: Train QMIA excluding certain classes, evaluate on both seen and unseen
2. **Baseline Comparison**: Compare QMIA performance against simple baseline attack
3. **Multiple Metrics**: AUC and TPR at 1% FPR for comprehensive evaluation
4. **Reusable Base Models**: Train base model once, reuse for multiple QMIA experiments
5. **Comprehensive Logging**: Detailed results saved with verbose filenames

## Output Structure

```
tabular/
├── models/
│   ├── texas_base_model/
│   │   ├── model.pth
│   │   ├── preprocessors.pth
│   │   └── texas_predictions.csv
│   └── purchase_base_model/
│       ├── model.pth
│       ├── preprocessors.pth
│       └── purchase_predictions.csv
└── logs/
    ├── texas_qmia_summary_dropped_90_91_92_93_94_seed_42.csv
    ├── texas_qmia_dropped_90_91_92_93_94_seed_42.csv
    ├── texas_results_table.csv
    └── texas_results_plot.png
```

## Parameters

### Base Training
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 256)
- `--lr`: Learning rate (default: 0.001)

### QMIA Training
- `--dropped_classes`: List of class IDs to drop (exclude from QMIA training)
- `--seed`: Random seed for reproducibility

### Runner
- `--dropped_ratios`: Fractions of classes to drop from QMIA training
- `--base_only`: Only train base model, skip QMIA
- `--skip_base_training`: Skip base training even if model doesn't exist

## Example Results

The scripts measure:
- **Overall Performance**: QMIA vs baseline on all samples
- **Seen Classes**: Performance on classes used to train QMIA
- **Unseen Classes**: Performance on classes dropped from QMIA training

Key research question: How does QMIA performance degrade on unseen classes compared to seen classes?

## Dependencies

- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- catboost
- optuna 