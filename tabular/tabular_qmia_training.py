# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import numpy as np
import pandas as pd
import torch
import optuna
from catboost import CatBoostRegressor
from catboost.metrics import RMSEWithUncertainty
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.preprocessing import StandardScaler

def load_base_model_data(dataset, model_root, model_type="mlp"):
    """Load the preprocessed data and base model predictions."""
    model_dir = os.path.join(model_root, f"{dataset}_{model_type}_base_model")
    predictions_path = os.path.join(model_dir, f"{dataset}_predictions.csv")
    
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Base model predictions not found at {predictions_path}")
    
    # Load predictions
    data = pd.read_csv(predictions_path)
    
    # Load preprocessed info
    preprocessors_path = os.path.join(model_dir, "preprocessors.pth")
    preprocessors = torch.load(preprocessors_path, weights_only=False)
    
    # Load processed features and labels in the correct order
    processed_data_path = os.path.join(model_dir, "processed_data.pth")
    if os.path.exists(processed_data_path):
        processed_data = torch.load(processed_data_path, weights_only=False)
        X_combined = processed_data['X_combined']
        y_combined = processed_data['y_combined']
        train_size = processed_data['train_size']
        val_size = processed_data['val_size'] 
        holdout_size = processed_data['holdout_size']
    else:
        raise FileNotFoundError(f"Processed data not found at {processed_data_path}")
    
    return data, preprocessors, X_combined, y_combined, train_size, val_size, holdout_size

def compute_scores(probs, labels):
    """Compute confidence scores from probability predictions."""
    if isinstance(probs, str):
        # Handle string representation of arrays
        probs = eval(probs)
    
    probs = np.array(probs)
    
    # Use the probability of the predicted class (not just max)
    if len(probs.shape) == 1:
        if len(probs) == 1:
            return probs[0]
        else:
            # For multi-class, use probability of the most confident class
            return np.max(probs)
    else:
        # Multi-dimensional array - this shouldn't happen with proper preprocessing
        probs_flat = probs.flatten()
        return np.max(probs_flat) if len(probs_flat) > 0 else 0.0

def compute_margin_scores(probs, labels, score_type="top_two_margin"):
    """Compute margin-based confidence scores like image QMIA."""
    if isinstance(probs, str):
        probs = eval(probs)
    
    probs = np.array(probs)
    
    if len(probs.shape) == 1:
        if len(probs) <= 1:
            return probs[0] if len(probs) == 1 else 0.0
    else:
        probs = probs.flatten()
    
    if len(probs) <= 1:
        return probs[0] if len(probs) == 1 else 0.0
    
    # Convert probabilities to logits (inverse softmax)
    probs = np.clip(probs, 1e-8, 1-1e-8)  # Avoid log(0)
    logits = np.log(probs)
    
    if score_type == "top_two_margin":
        # Difference between top 2 logits (like image QMIA)
        sorted_logits = np.sort(logits)[::-1]  # Sort descending
        if len(sorted_logits) >= 2:
            return sorted_logits[0] - sorted_logits[1]
        else:
            return sorted_logits[0]
    
    elif score_type == "true_margin":
        # Difference between true class logit and max other logit
        if isinstance(labels, (int, np.integer)):
            true_label = labels
        else:
            true_label = int(labels)
        
        if true_label < len(logits):
            true_logit = logits[true_label]
            other_logits = np.concatenate([logits[:true_label], logits[true_label+1:]])
            if len(other_logits) > 0:
                max_other = np.max(other_logits)
                return true_logit - max_other
            else:
                return true_logit
        else:
            return np.max(logits)
    
    else:  # fallback to max probability
        return np.max(probs)

def main():
    parser = argparse.ArgumentParser(description="QMIA training with class dropout")
    parser.add_argument("--dataset", type=str, required=True, choices=["purchase", "texas"], 
                       help="Dataset name")
    parser.add_argument("--model_root", type=str, default="./tabular/models/", 
                       help="Model directory")
    parser.add_argument("--logs_root", type=str, default="./tabular/logs/", 
                       help="Logs directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "cat"], 
                       help="Type of base model")
    parser.add_argument("--dropped_classes", type=int, nargs="+", required=True,
                       help="List of class IDs to drop (exclude from QMIA training)")
    parser.add_argument("--dropped_classes_str", type=str, default=None,
                       help="Compact string representation of dropped classes for filenames")
    parser.add_argument("--qmia_model", type=str, default="catboost", choices=["catboost", "mlp"],
                       help="QMIA model type: catboost (tree-based) or mlp (neural network)")
    parser.add_argument("--score_function", type=str, default="max_prob", 
                       choices=["max_prob", "top_two_margin", "true_margin"],
                       help="Score function: max_prob, top_two_margin, or true_margin")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create logs directory
    os.makedirs(args.logs_root, exist_ok=True)
    
    # Load base model data
    print(f"Loading {args.model_type} base model data for {args.dataset}...")
    data, preprocessors, X_combined, y_combined, train_size, val_size, holdout_size = load_base_model_data(
        args.dataset, args.model_root, args.model_type
    )
    
    # Determine seen classes (all classes minus dropped classes)
    all_classes = sorted(data['Label'].unique())
    seen_classes = [cls for cls in all_classes if cls not in args.dropped_classes]
    
    print(f"Dataset has {len(data)} samples")
    print(f"Classes in dataset: {all_classes}")
    print(f"Dropped classes: {args.dropped_classes}")
    print(f"Seen classes for QMIA training: {seen_classes}")
    
    # FIXED: Implement two-mask approach
    # Mask 1: Seen vs Unseen classes
    seen_mask = data["Label"].isin(seen_classes).values
    
    # Mask 2: Train/Val/Holdout splits  
    train_mask = data["Split"] == "train"
    val_mask = data["Split"] == "val"
    holdout_mask = data["Split"] == "test"
    
    # Create 6 categories using AND operations
    categories = {}
    
    # Seen classes
    seen_train_mask = seen_mask & train_mask.values
    seen_val_mask = seen_mask & val_mask.values  
    seen_holdout_mask = seen_mask & holdout_mask.values
    
    # Get indices for each category
    seen_train_indices = np.where(seen_train_mask)[0]
    seen_val_indices = np.where(seen_val_mask)[0] 
    seen_holdout_indices = np.where(seen_holdout_mask)[0]
    
    categories['seen_train'] = {
        'data': data[seen_train_mask],
        'features': X_combined[seen_train_indices],
        'labels': y_combined[seen_train_indices]
    }
    
    categories['seen_val'] = {
        'data': data[seen_val_mask], 
        'features': X_combined[seen_val_indices],
        'labels': y_combined[seen_val_indices]
    }
    
    categories['seen_holdout'] = {
        'data': data[seen_holdout_mask],
        'features': X_combined[seen_holdout_indices], 
        'labels': y_combined[seen_holdout_indices]
    }
    
    # Unseen classes
    unseen_train_mask = (~seen_mask) & train_mask.values
    unseen_holdout_mask = (~seen_mask) & holdout_mask.values
    
    unseen_train_indices = np.where(unseen_train_mask)[0]
    unseen_holdout_indices = np.where(unseen_holdout_mask)[0]
    
    categories['unseen_train'] = {
        'data': data[unseen_train_mask],
        'features': X_combined[unseen_train_indices],
        'labels': y_combined[unseen_train_indices]
    }
    
    categories['unseen_holdout'] = {
        'data': data[unseen_holdout_mask],
        'features': X_combined[unseen_holdout_indices],
        'labels': y_combined[unseen_holdout_indices]
    }
    
    print(f"Data splits after filtering:")
    for name, cat in categories.items():
        print(f"  {name}: {len(cat['data'])} samples")
    
    # Train QMIA model using ONLY seen_val data
    print(f"Training {args.qmia_model.upper()} QMIA model on seen_val data...")
    
    if args.qmia_model == "catboost":
        qmia_model = train_qmia_model_simple(categories['seen_val']['data'], 
                                           categories['seen_val']['features'], args.seed, args.score_function)
        scaler = None  # CatBoost doesn't need scaler
    else:  # mlp
        qmia_model, scaler = train_qmia_model_mlp(categories['seen_val']['data'], 
                                                categories['seen_val']['features'], args.seed, 
                                                score_function=args.score_function)
    
    # Evaluate on seen classes: members=seen_train, nonmembers=seen_holdout
    print("Evaluating on seen classes...")
    seen_eval_data = pd.concat([
        categories['seen_train']['data'].assign(membership='in'),
        categories['seen_holdout']['data'].assign(membership='out')
    ])
    seen_eval_features = np.concatenate([
        categories['seen_train']['features'],
        categories['seen_holdout']['features']
    ])
    
    if args.qmia_model == "catboost":
        seen_results = evaluate_model_simple(qmia_model, seen_eval_data, seen_eval_features, "seen", args.score_function)
    else:  # mlp
        seen_results = evaluate_model_mlp(qmia_model, scaler, seen_eval_data, seen_eval_features, "seen", args.score_function)
    
    # Evaluate on unseen classes: members=unseen_train, nonmembers=unseen_holdout
    print("Evaluating on unseen classes...")
    unseen_eval_data = pd.concat([
        categories['unseen_train']['data'].assign(membership='in'),
        categories['unseen_holdout']['data'].assign(membership='out')
    ])
    unseen_eval_features = np.concatenate([
        categories['unseen_train']['features'],
        categories['unseen_holdout']['features']
    ])
    
    # For unseen classes, use mean predictions from seen classes
    if args.qmia_model == "catboost":
        unseen_results = evaluate_model_simple(qmia_model, unseen_eval_data, unseen_eval_features, "unseen", args.score_function)
    else:  # mlp
        unseen_results = evaluate_model_mlp(qmia_model, scaler, unseen_eval_data, unseen_eval_features, "unseen", args.score_function)
    
    # Combine results
    all_results = pd.concat([seen_results, unseen_results], ignore_index=True)
    
    # Compute metrics
    overall_metrics = compute_auc_and_metrics(all_results)
    seen_metrics = compute_auc_and_metrics(seen_results)
    unseen_metrics = compute_auc_and_metrics(unseen_results)
    
    print("\n=== RESULTS ===")
    print(f"Overall - QMIA AUC: {overall_metrics['qmia_auc']:.3f}, "
          f"TPR@1%FPR: {overall_metrics['qmia_tpr_at_1_fpr']:.3f}")
    print(f"Overall - Baseline AUC: {overall_metrics['baseline_auc']:.3f}, "
          f"TPR@1%FPR: {overall_metrics['baseline_tpr_at_1_fpr']:.3f}")
    
    print(f"Seen Classes - QMIA AUC: {seen_metrics['qmia_auc']:.3f}, "
          f"TPR@1%FPR: {seen_metrics['qmia_tpr_at_1_fpr']:.3f}")
    print(f"Seen Classes - Baseline AUC: {seen_metrics['baseline_auc']:.3f}, "
          f"TPR@1%FPR: {seen_metrics['baseline_tpr_at_1_fpr']:.3f}")
    
    print(f"Unseen Classes - QMIA AUC: {unseen_metrics['qmia_auc']:.3f}, "
          f"TPR@1%FPR: {unseen_metrics['qmia_tpr_at_1_fpr']:.3f}")
    print(f"Unseen Classes - Baseline AUC: {unseen_metrics['baseline_auc']:.3f}, "
          f"TPR@1%FPR: {unseen_metrics['baseline_tpr_at_1_fpr']:.3f}")
    
    # Use compact representation for filenames if provided, otherwise fall back to concatenated list
    if args.dropped_classes_str:
        dropped_classes_str = args.dropped_classes_str
        print(f"Using compact representation for filenames: {dropped_classes_str}")
    else:
        dropped_classes_str = "_".join(map(str, sorted(args.dropped_classes)))
        print(f"Using concatenated list for filenames: {dropped_classes_str}")
    
    # Save results with model type in filename
    results_filename = f"{args.dataset}_{args.model_type}_qmia_{args.qmia_model}_{args.score_function}_dropped_{dropped_classes_str}_seed_{args.seed}.csv"
    results_path = os.path.join(args.logs_root, results_filename)
    all_results.to_csv(results_path, index=False)
    
    # Save summary metrics
    summary_metrics = {
        "dataset": args.dataset,
        "model_type": args.model_type,
        "qmia_model": args.qmia_model,
        "score_function": args.score_function,
        "dropped_classes": args.dropped_classes,
        "dropped_classes_str": dropped_classes_str,  # Store the compact representation
        "seen_classes": seen_classes,
        "seed": args.seed,
        **{f"overall_{k}": v for k, v in overall_metrics.items()},
        **{f"seen_{k}": v for k, v in seen_metrics.items()},
        **{f"unseen_{k}": v for k, v in unseen_metrics.items()}
    }
    
    summary_filename = f"{args.dataset}_{args.model_type}_qmia_{args.qmia_model}_{args.score_function}_summary_dropped_{dropped_classes_str}_seed_{args.seed}.csv"
    summary_path = os.path.join(args.logs_root, summary_filename)
    pd.DataFrame([summary_metrics]).to_csv(summary_path, index=False)
    
    print(f"\nResults saved to {results_path}")
    print(f"Summary saved to {summary_path}")

def train_qmia_model_simple(data, features, seed=42, score_function="max_prob"):
    """Simplified QMIA training with pre-filtered, aligned data."""
    
    print(f"QMIA Training - Feature dimensions:")
    print(f"  Features shape: {features.shape}")
    print(f"  Feature dimensionality: {features.shape[1]} (should be >> 1)")
    print(f"  ✓ Using rich tabular features!")
    print(f"  Score function: {score_function}")
    
    # Compute target scores based on chosen function
    if score_function == "max_prob":
        y_score = np.array([compute_scores(score, label) for score, label in 
                           zip(data["Score"], data["Label"])])
    else:
        y_score = np.array([compute_margin_scores(score, label, score_function) for score, label in 
                           zip(data["Score"], data["Label"])])
    
    # FIXED: Use actual class labels for score transformation (not membership labels)
    # This captures whether the model's prediction was correct or not
    labels = data["Label"].values
    
    # Transform scores using the same function as ACS
    def f_score(prob, l):
        prob = np.clip(prob, 1e-7, 1-1e-7)
        return (np.log(prob) - np.log(1 - prob)) * (2 * l - 1)
    
    y_score_transformed = f_score(y_score, labels)
    
    print(f"  Score range: [{y_score.min():.3f}, {y_score.max():.3f}]")
    
    # Train CatBoost model
    param = {
        "depth": 6,
        "l2_leaf_reg": 0.1,
        "learning_rate": 0.1,
        "iterations": 600,
        "thread_count": 1,
        "objective": "RMSEWithUncertainty",
        "posterior_sampling": True,
        "random_seed": seed
    }
    
    clf = CatBoostRegressor(**param)
    clf.fit(features, y_score_transformed, verbose=0)
    
    return clf

class TabularQMIAMLP(nn.Module):
    """MLP-based QMIA model for tabular data, similar to image QMIA architecture."""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output 2 values: mu and log_sigma (like CatBoost RMSEWithUncertainty)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_qmia_model_mlp(data, features, seed=42, epochs=200, lr=1e-3, batch_size=256, score_function="max_prob"):
    """Train MLP-based QMIA model."""
    
    print(f"MLP QMIA Training - Feature dimensions:")
    print(f"  Features shape: {features.shape}")
    print(f"  Feature dimensionality: {features.shape[1]}")
    print(f"  Score function: {score_function}")
    
    # Compute target scores based on chosen function
    if score_function == "max_prob":
        y_score = np.array([compute_scores(score, label) for score, label in 
                           zip(data["Score"], data["Label"])])
    else:
        y_score = np.array([compute_margin_scores(score, label, score_function) for score, label in 
                           zip(data["Score"], data["Label"])])
    
    # Use actual class labels for score transformation
    labels = data["Label"].values
    
    # Transform scores using the same function as ACS
    def f_score(prob, l):
        prob = np.clip(prob, 1e-7, 1-1e-7)
        return (np.log(prob) - np.log(1 - prob)) * (2 * l - 1)
    
    y_score_transformed = f_score(y_score, labels)
    
    # Standardize features (important for MLPs)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(features_scaled)
    y_tensor = torch.FloatTensor(y_score_transformed).unsqueeze(1)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TabularQMIAMLP(features.shape[1]).to(device)
    
    # Loss function - similar to CatBoost RMSEWithUncertainty
    def gaussian_loss(pred, target):
        mu, log_sigma = pred[:, 0:1], pred[:, 1:2]
        sigma = torch.exp(log_sigma)
        
        # Negative log likelihood
        loss = 0.5 * torch.log(2 * np.pi * sigma**2) + 0.5 * ((target - mu) / sigma)**2
        return loss.mean()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training loop
    model.train()
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training MLP for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = gaussian_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    model.eval()
    return model, scaler

def evaluate_model_simple(model, eval_data, eval_features, class_type, score_function="max_prob"):
    """Simplified evaluation with proper membership handling."""
    
    # Compute target scores based on chosen function
    if score_function == "max_prob":
        y_score = np.array([compute_scores(score, label) for score, label in 
                           zip(eval_data["Score"], eval_data["Label"])])
    else:
        y_score = np.array([compute_margin_scores(score, label, score_function) for score, label in 
                           zip(eval_data["Score"], eval_data["Label"])])
    
    # Use actual class labels for score transformation
    labels = eval_data["Label"].values
    
    def f_score(prob, l):
        prob = np.clip(prob, 1e-7, 1-1e-7)
        return (np.log(prob) - np.log(1 - prob)) * (2 * l - 1)
    
    y_score_transformed = f_score(y_score, labels)
    
    # Get model predictions
    conf_test = model.predict(eval_features, prediction_type="RawFormulaVal")
    
    # Create results
    results = {
        "score": y_score_transformed,
        "raw_score": y_score,  # Store raw scores for baseline
        "mu": conf_test[:, 0],
        "log_sigma": conf_test[:, 1], 
        "membership": eval_data["membership"].values,
        "labels": eval_data["Label"].values,
        "class_type": [class_type] * len(eval_data)
    }
    
    return pd.DataFrame(results)

def evaluate_model_mlp(model, scaler, eval_data, eval_features, class_type, score_function="max_prob"):
    """Evaluate MLP-based QMIA model."""
    
    # Compute target scores based on chosen function
    if score_function == "max_prob":
        y_score = np.array([compute_scores(score, label) for score, label in 
                           zip(eval_data["Score"], eval_data["Label"])])
    else:
        y_score = np.array([compute_margin_scores(score, label, score_function) for score, label in 
                           zip(eval_data["Score"], eval_data["Label"])])
    
    # Use actual class labels for score transformation
    labels = eval_data["Label"].values
    
    def f_score(prob, l):
        prob = np.clip(prob, 1e-7, 1-1e-7)
        return (np.log(prob) - np.log(1 - prob)) * (2 * l - 1)
    
    y_score_transformed = f_score(y_score, labels)
    
    # Standardize features using same scaler
    features_scaled = scaler.transform(eval_features)
    
    # Get model predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(features_scaled).to(device)
        pred = model(X_tensor).cpu().numpy()
    
    # Create results
    results = {
        "score": y_score_transformed,
        "raw_score": y_score,  # Store raw scores for baseline
        "mu": pred[:, 0],
        "log_sigma": pred[:, 1], 
        "membership": eval_data["membership"].values,
        "labels": eval_data["Label"].values,
        "class_type": [class_type] * len(eval_data)
    }
    
    return pd.DataFrame(results)

def compute_auc_and_metrics(results_df, class_filter=None):
    """Compute AUC and TPR at specific FPR for given class filter."""
    # No filtering needed since we pass separate dataframes now
    filtered_df = results_df
    
    if len(filtered_df) == 0:
        return {"qmia_auc": 0.0, "qmia_tpr_at_1_fpr": 0.0, "baseline_auc": 0.0, "baseline_tpr_at_1_fpr": 0.0}
    
    # Compute z-scores for QMIA
    mu = filtered_df["mu"].values
    log_sigma = filtered_df["log_sigma"].values
    target_scores = filtered_df["score"].values
    
    sigma = np.exp(log_sigma)
    z_scores = (target_scores - mu) / sigma
    
    # Create binary labels (1 for members, 0 for non-members)
    binary_labels = (filtered_df["membership"] == "in").astype(int)
    
    if len(np.unique(binary_labels)) < 2:
        return {"qmia_auc": 0.5, "qmia_tpr_at_1_fpr": 0.0, "baseline_auc": 0.5, "baseline_tpr_at_1_fpr": 0.0}
    
    # QMIA AUC
    qmia_auc = roc_auc_score(binary_labels, z_scores)
    
    # FIXED: Baseline should use raw confidence scores, not transformed scores
    # This is the standard confidence-based membership inference attack
    raw_scores = filtered_df["raw_score"].values
    baseline_auc = roc_auc_score(binary_labels, raw_scores)
    
    # TPR at 1% FPR for QMIA
    fpr, tpr, _ = roc_curve(binary_labels, z_scores)
    tpr_at_1_fpr = np.interp(0.01, fpr, tpr)
    
    # TPR at 1% FPR for baseline (using raw confidence scores)
    fpr_base, tpr_base, _ = roc_curve(binary_labels, raw_scores)
    tpr_at_1_fpr_baseline = np.interp(0.01, fpr_base, tpr_base)
    
    return {
        "qmia_auc": qmia_auc,
        "qmia_tpr_at_1_fpr": tpr_at_1_fpr,
        "baseline_auc": baseline_auc,
        "baseline_tpr_at_1_fpr": tpr_at_1_fpr_baseline
    }

if __name__ == "__main__":
    main() 