#!/usr/bin/env python3
"""
Convert the Texas dataset from binary format into ImageNet-style directory structure.
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

def load_texas_data(data_root):
    """Load Texas dataset from binary files."""
    feats_path = os.path.join(data_root, "texas", "100", "feats")
    labels_path = os.path.join(data_root, "texas", "100", "labels")
    
    if not os.path.exists(feats_path):
        raise FileNotFoundError(f"Features file not found: {feats_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    # Load features: support CSV text or binary
    # Heuristic: if the header contains commas/newlines and is ASCII -> CSV
    is_text_csv = False
    with open(feats_path, 'rb') as f:
        head = f.read(4096)
        try:
            head_decoded = head.decode('ascii', errors='ignore')
            if (',' in head_decoded) or ('\n' in head_decoded) or ('\r' in head_decoded):
                is_text_csv = True
        except Exception:
            is_text_csv = False

    if is_text_csv:
        try:
            features = np.loadtxt(feats_path, delimiter=',', dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV features from {feats_path}: {e}")
    else:
        # Binary fallback (original behavior)
        features = np.fromfile(feats_path, dtype=np.float32)
    
    # Load labels from text file (not binary!)
    with open(labels_path, 'r') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            labels.append(int(line))
    labels = np.array(labels, dtype=np.int32)
    
    # Calculate number of features based on actual data
    n_samples = len(labels)
    if features.ndim == 2:
        # Ensure rows correspond to samples
        if features.shape[0] != n_samples and features.shape[1] == n_samples:
            features = features.T
        if features.shape[0] != n_samples:
            raise ValueError(f"Features shape {features.shape} incompatible with labels length {n_samples}")
        n_features = features.shape[1]
    else:
        n_features = len(features) // n_samples
    
    # Ensure we have the exact right number of elements
    if features.ndim == 1:
        total_expected = n_samples * n_features
        if len(features) > total_expected:
            print(f"Warning: Truncating features from {len(features)} to {total_expected} elements")
            features = features[:total_expected]
        elif len(features) < total_expected:
            raise ValueError(f"Not enough features: have {len(features)}, need {total_expected}")
        # Reshape features
        features = features.reshape(n_samples, n_features)

    # Ensure dtype
    if features.dtype != np.float32:
        features = features.astype(np.float32, copy=False)
    
    print(f"Loaded Texas dataset: {n_samples} samples, {n_features} features, {len(np.unique(labels))} classes")
    print(f"Feature shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: {labels.min()}-{labels.max()}")
    
    return features, labels

def convert_texas_dataset(data_root="./data", num_workers=32):
    """
    Convert Texas dataset from binary format to directory structure:
    
    data/texas/
    ├── trainval/
    │   ├── class_1/
    │   │   ├── 00000000.npy
    │   │   ├── 00000001.npy
    │   │   └── ...
    │   ├── class_2/
    │   └── ...
    └── test/
        ├── class_1/
        └── ...
    """
    
    # Load the original dataset
    print("Loading original Texas dataset...")
    features, labels = load_texas_data(data_root)
    
    # Compute dataset statistics
    print("\nComputing dataset statistics...")
    dataset_mean = np.mean(features, axis=0)  # Mean per feature
    dataset_std = np.std(features, axis=0)    # Std per feature
    
    # For tabular data, we often use overall mean/std across all features
    overall_mean = np.mean(dataset_mean)
    overall_std = np.mean(dataset_std)
    
    print(f"Dataset statistics:")
    print(f"  Feature-wise mean range: {dataset_mean.min():.6f} to {dataset_mean.max():.6f}")
    print(f"  Feature-wise std range:  {dataset_std.min():.6f} to {dataset_std.max():.6f}")
    print(f"  Overall mean: {overall_mean:.6f}")
    print(f"  Overall std:  {overall_std:.6f}")
    
    print(f"\n📋 Add to DATASET_FLAGS in data_utils.py:")
    print(f"    TEXAS_MEAN = ({overall_mean:.6f},)")
    print(f"    TEXAS_STD = ({overall_std:.6f},)")
    
    # Check if labels are 0-based or 1-based and adjust accordingly
    min_label = labels.min()
    max_label = labels.max()
    
    if min_label == 0:
        # Labels are 0-based, convert to 1-based for consistency with Purchase dataset
        print(f"Converting 0-based labels ({min_label}-{max_label}) to 1-based")
        labels = labels + 1
    elif min_label == 1:
        # Labels are already 1-based, keep as-is
        print(f"Labels are already 1-based ({min_label}-{max_label}), keeping as-is")
    else:
        # Unexpected label range
        print(f"Warning: Unexpected label range ({min_label}-{max_label})")
    
    # Combine features and labels into data list
    data = [(features[i], labels[i]) for i in range(len(features))]
    
    print(f"Loaded {len(data)} samples")
    
    # Create output directory structure
    output_dir = Path(data_root) / "texas"
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
    convert_texas_dataset() 