# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
import sys
from itertools import combinations
import re

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    # Stream output in real-time instead of capturing
    result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"\nSUCCESS: {description} completed")
        return True

def parse_class_specification(class_spec_list):
    """Parse class specifications that can include ranges like '10-99'."""
    expanded_classes = []
    compact_representation = []
    
    for spec in class_spec_list:
        if isinstance(spec, int):
            # Regular integer
            expanded_classes.append(spec)
            compact_representation.append(str(spec))
        elif isinstance(spec, str) and '-' in spec:
            # Range specification like "10-99"
            try:
                start, end = map(int, spec.split('-'))
                if start > end:
                    raise ValueError(f"Invalid range: {spec} (start > end)")
                range_classes = list(range(start, end + 1))
                expanded_classes.extend(range_classes)
                compact_representation.append(spec)  # Keep original range string
                print(f"Expanded range {spec} to {len(range_classes)} classes: {start}...{end}")
            except ValueError as e:
                print(f"Error parsing range '{spec}': {e}")
                raise
        else:
            # Try to convert string to int
            try:
                class_id = int(spec)
                expanded_classes.append(class_id)
                compact_representation.append(str(class_id))
            except ValueError:
                raise ValueError(f"Invalid class specification: {spec}")
    
    # Create compact string representation
    compact_str = "_".join(compact_representation)
    
    return expanded_classes, compact_str

def check_base_model_exists(dataset, model_root, model_type="mlp"):
    """Check if base model already exists."""
    model_dir = os.path.join(model_root, f"{dataset}_{model_type}_base_model")
    predictions_file = os.path.join(model_dir, f"{dataset}_predictions.csv")
    processed_data_file = os.path.join(model_dir, "processed_data.pth")
    
    if model_type == "mlp":
        model_file = os.path.join(model_dir, "model.pth")
    elif model_type == "cat":
        model_file = os.path.join(model_dir, "model.cbm")
    else:
        return False
    
    return (os.path.exists(model_file) and 
            os.path.exists(predictions_file) and 
            os.path.exists(processed_data_file))

def get_dataset_classes(dataset):
    """Get the number of classes for each dataset."""
    if dataset == "purchase":
        return list(range(100))  # Purchase-100 has 100 classes
    elif dataset == "texas":
        return list(range(100))  # Texas-100 has 100 classes
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def generate_class_scenarios(all_classes, dropped_ratios=[0.2, 0.5, 0.8]):
    """Generate different scenarios of dropped class splits."""
    scenarios = []
    
    for ratio in dropped_ratios:
        num_dropped = int(len(all_classes) * ratio)
        # Take last num_dropped classes as dropped classes  
        dropped_classes = all_classes[-num_dropped:]
        
        # Create compact representation for ratio-based scenarios
        if num_dropped > 10:  # Use range notation for large ranges
            start_class = dropped_classes[0]
            end_class = dropped_classes[-1]
            compact_str = f"{start_class}-{end_class}"
        else:
            compact_str = "_".join(map(str, dropped_classes))
        
        scenarios.append((dropped_classes, compact_str))
    
    return scenarios

def main():
    parser = argparse.ArgumentParser(description="Run tabular QMIA experiments")
    parser.add_argument("--dataset", type=str, required=True, choices=["purchase", "texas"], 
                       help="Dataset name")
    parser.add_argument("--data_root", type=str, default="./data/", help="Data root directory")
    parser.add_argument("--model_root", type=str, default="./tabular/models/", help="Model directory")
    parser.add_argument("--logs_root", type=str, default="./tabular/logs/", help="Logs directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "cat"], 
                       help="Type of base model")
    
    # Base model training parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for base model")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for base model")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for base model")
    
    # QMIA experiment parameters
    parser.add_argument("--dropped_ratios", type=float, nargs="+", default=[0.2, 0.5, 0.8],
                       help="Ratios of classes to drop (exclude from QMIA training)")
    parser.add_argument("--custom_dropped_classes", type=str, nargs="*", default=None,
                       help="Custom list of dropped classes (supports ranges like '10-99' or individual classes)")
    
    # Control flags
    parser.add_argument("--skip_base_training", action="store_true", 
                       help="Skip base model training even if model doesn't exist")
    parser.add_argument("--base_only", action="store_true", 
                       help="Only run base model training, skip QMIA experiments")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.model_root, exist_ok=True)
    os.makedirs(args.logs_root, exist_ok=True)
    
    print(f"Running experiments for {args.dataset} dataset with {args.model_type} base model")
    print(f"Model root: {args.model_root}")
    print(f"Logs root: {args.logs_root}")
    
    # Step 1: Train base model if needed
    if not args.skip_base_training:
        if check_base_model_exists(args.dataset, args.model_root, args.model_type):
            print(f"\nBase model for {args.dataset} ({args.model_type}) already exists. Skipping training.")
        else:
            print(f"\nTraining {args.model_type} base model for {args.dataset}...")
            base_cmd = [
                sys.executable, "tabular/tabular_base_training.py",
                "--dataset", args.dataset,
                "--data_root", args.data_root,
                "--model_root", args.model_root,
                "--seed", str(args.seed),
                "--model_type", args.model_type,
                "--batch_size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--lr", str(args.lr)
            ]
            
            if not run_command(base_cmd, f"{args.model_type.upper()} base model training for {args.dataset}"):
                print("Base model training failed. Exiting.")
                return
    else:
        print("Skipping base model training as requested.")
    
    if args.base_only:
        print("Base-only mode. Exiting after base model training.")
        return
    
    # Step 2: Run QMIA experiments
    if not check_base_model_exists(args.dataset, args.model_root, args.model_type):
        print(f"ERROR: {args.model_type.upper()} base model does not exist. Cannot run QMIA experiments.")
        print("Please run base training first or check the model directory.")
        return
    
    # Generate class scenarios
    all_classes = get_dataset_classes(args.dataset)
    
    if args.custom_dropped_classes is not None:
        # Parse custom dropped classes (supports ranges)
        try:
            expanded_classes, compact_str = parse_class_specification(args.custom_dropped_classes)
            scenarios = [(expanded_classes, compact_str)]
            print(f"\nUsing custom dropped classes: {args.custom_dropped_classes}")
            print(f"Expanded to {len(expanded_classes)} classes: {expanded_classes[:10]}{'...' if len(expanded_classes) > 10 else ''}")
            print(f"Will save with compact representation: {compact_str}")
        except ValueError as e:
            print(f"Error in custom dropped classes specification: {e}")
            return
    else:
        scenarios = generate_class_scenarios(all_classes, args.dropped_ratios)
        print(f"\nGenerated {len(scenarios)} class scenarios:")
        for i, (dropped_classes, compact_str) in enumerate(scenarios):
            ratio = len(dropped_classes) / len(all_classes)
            num_seen = len(all_classes) - len(dropped_classes)
            print(f"  Scenario {i+1}: Drop {len(dropped_classes)} classes ({ratio:.1%}), Keep {num_seen} classes")
            print(f"    Classes: {dropped_classes[:5]}{'...' if len(dropped_classes) > 5 else ''}")
            print(f"    Compact: {compact_str}")
    
    # Run QMIA experiments for each scenario
    print(f"\nRunning QMIA experiments...")
    successful_experiments = 0
    total_experiments = len(scenarios)
    
    for i, (dropped_classes, compact_str) in enumerate(scenarios):
        print(f"\n{'-'*80}")
        print(f"EXPERIMENT {i+1}/{total_experiments}")
        print(f"Dropped classes: {dropped_classes[:10]}{'...' if len(dropped_classes) > 10 else ''} ({len(dropped_classes)} total)")
        print(f"Compact representation: {compact_str}")
        print(f"{'-'*80}")
        
        qmia_cmd = [
            sys.executable, "tabular/tabular_qmia_training.py",
            "--dataset", args.dataset,
            "--model_root", args.model_root,
            "--logs_root", args.logs_root,
            "--seed", str(args.seed),
            "--model_type", args.model_type,
            "--dropped_classes_str", compact_str,
            "--dropped_classes"
        ] + [str(cls) for cls in dropped_classes]
        
        if run_command(qmia_cmd, f"QMIA experiment {i+1}"):
            successful_experiments += 1
        else:
            print(f"QMIA experiment {i+1} failed. Continuing with next experiment.")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Base model type: {args.model_type}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful experiments: {successful_experiments}")
    print(f"Failed experiments: {total_experiments - successful_experiments}")
    
    if successful_experiments > 0:
        print(f"\nResults saved in: {args.logs_root}")
        print("Look for files matching:")
        print(f"  - {args.dataset}_{args.model_type}_qmia_summary_dropped_*_seed_{args.seed}.csv")
        print(f"  - {args.dataset}_{args.model_type}_qmia_dropped_*_seed_{args.seed}.csv")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 