#!/usr/bin/env python3
"""
Convert the monolithic Purchase dataset into ImageNet-style directory structure.
This avoids concurrent file access issues during training.
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def save_sample(args):
    """Helper function to save a single sample (for multiprocessing)"""
    features, label, filepath = args
    np.save(filepath, features)
    return True

def convert_purchase_dataset(data_root="./data", num_workers=32):
    """
    Convert dataset_purchase from single file to directory structure:
    
    data/purchase/
    ├── trainval/
    │   ├── class_0/
    │   │   ├── 00000000.npy
    │   │   ├── 00000001.npy
    │   │   └── ...
    │   ├── class_1/
    │   └── ...
    └── test/
        ├── class_0/
        └── ...
    """
    
    # Load the original dataset
    original_path = os.path.join(data_root, "dataset_purchase")
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original dataset not found at {original_path}")
    
    print("Loading original Purchase dataset...")
    data = []
    with open(original_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > 1:  # Skip empty lines
                label = int(parts[0])
                features = [int(x) for x in parts[1:]]
                data.append((np.array(features, dtype=np.float32), label))
    
    print(f"Loaded {len(data)} samples")
    
    # Create output directory structure
    output_dir = Path(data_root) / "purchase"
    trainval_dir = output_dir / "trainval"
    test_dir = output_dir / "test"
    
    # Get unique classes
    classes = sorted(set(item[1] for item in data))
    print(f"Found {len(classes)} classes: {min(classes)} to {max(classes)}")
    
    # Create class directories
    for class_id in classes:
        (trainval_dir / f"class_{class_id}").mkdir(parents=True, exist_ok=True)
        (test_dir / f"class_{class_id}").mkdir(parents=True, exist_ok=True)
    
    # Split data: 80% trainval, 20% test (stratified by class)
    np.random.seed(42)  # For reproducible splits
    
    # Group data by class
    class_data = {}
    for idx, (features, label) in enumerate(data):
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(idx)
    
    trainval_indices = []
    test_indices = []
    
    # Stratified split for each class
    for class_id, class_indices in class_data.items():
        class_indices = np.array(class_indices)
        np.random.shuffle(class_indices)  # Shuffle within class
        
        split_point = int(0.8 * len(class_indices))
        trainval_indices.extend(class_indices[:split_point])
        test_indices.extend(class_indices[split_point:])
    
    # Shuffle the final indices to mix classes
    np.random.shuffle(trainval_indices)
    np.random.shuffle(test_indices)
    
    print(f"Stratified split: {len(trainval_indices)} trainval, {len(test_indices)} test")
    
    # Verify class distribution
    trainval_classes = [data[idx][1] for idx in trainval_indices]
    test_classes = [data[idx][1] for idx in test_indices]
    
    print("Class distribution verification:")
    for class_id in sorted(classes):
        trainval_count = sum(1 for c in trainval_classes if c == class_id)
        test_count = sum(1 for c in test_classes if c == class_id)
        total_count = trainval_count + test_count
        print(f"  Class {class_id}: {trainval_count}/{total_count} ({trainval_count/total_count:.1%}) trainval, {test_count}/{total_count} ({test_count/total_count:.1%}) test")
    
    # Prepare trainval data with pre-calculated filenames
    print("Preparing trainval data...")
    trainval_class_counters = {class_id: 0 for class_id in classes}
    trainval_args = []
    for idx in trainval_indices:
        features, label = data[idx]
        filename = f"{trainval_class_counters[label]:08d}.npy"
        filepath = trainval_dir / f"class_{label}" / filename
        trainval_args.append((features, label, filepath))
        trainval_class_counters[label] += 1
    
    print("Saving trainval data...")
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(save_sample, trainval_args), total=len(trainval_indices), desc="Trainval"))
    
    # Prepare test data with pre-calculated filenames
    print("Preparing test data...")
    test_class_counters = {class_id: 0 for class_id in classes}
    test_args = []
    for idx in test_indices:
        features, label = data[idx]
        filename = f"{test_class_counters[label]:08d}.npy"
        filepath = test_dir / f"class_{label}" / filename
        test_args.append((features, label, filepath))
        test_class_counters[label] += 1
    
    print("Saving test data...")
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(save_sample, test_args), total=len(test_indices), desc="Test"))
    
    print(f"✅ Conversion complete! Dataset saved to {output_dir}")
    
    # Print summary
    print("\nDataset summary:")
    for split in ["trainval", "test"]:
        split_dir = output_dir / split
        total_files = sum(len(list(class_dir.glob("*.npy"))) for class_dir in split_dir.iterdir())
        print(f"  {split}: {total_files} samples across {len(classes)} classes")

if __name__ == "__main__":
    convert_purchase_dataset() 