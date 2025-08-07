# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

class MLP(nn.Module):
    """Simple MLP for tabular data."""

    def __init__(self, in_shape, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_shape, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, inputs):
        """Forward pass of the model."""
        inputs = inputs.flatten(1)
        inputs = torch.tanh(self.fc1(inputs))
        outputs = self.fc2(inputs)
        return outputs

def load_dataset(dataset_name, data_root):
    """Load Purchase or Texas dataset."""
    if dataset_name == "purchase":
        # Load purchase dataset (CSV format)
        data_path = os.path.join(data_root, "dataset_purchase")
        
        # Read CSV file
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) > 1:  # Skip empty lines
                    label = int(parts[0])
                    features = [int(x) for x in parts[1:]]
                    data.append((features, label))
        
        # Convert to numpy arrays
        X = np.array([item[0] for item in data], dtype=np.float32)
        y = np.array([item[1] for item in data], dtype=np.int32)
        
    elif dataset_name == "texas":
        # Load texas dataset
        feats_path = os.path.join(data_root, "texas", "100", "feats")
        labels_path = os.path.join(data_root, "texas", "100", "labels")
        
        # Load features and labels - handle empty lines in labels file
        X = np.fromfile(feats_path, dtype=np.float32)
        
        # Load labels and filter out empty lines
        with open(labels_path, 'r') as f:
            lines = f.readlines()
        y = []
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                y.append(int(line))
        y = np.array(y, dtype=np.int32)
        
        # Calculate number of features based on actual data
        n_samples = len(y)
        n_features = len(X) // n_samples
        
        # Ensure we have the exact right number of elements
        total_expected = n_samples * n_features
        if len(X) > total_expected:
            print(f"Warning: Truncating features from {len(X)} to {total_expected} elements")
            X = X[:total_expected]
        elif len(X) < total_expected:
            raise ValueError(f"Not enough features: have {len(X)}, need {total_expected}")
        
        # Reshape features
        X = X.reshape(n_samples, n_features)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return X, y

def preprocess_data(X, y, model_type="mlp"):
    """Preprocess the data with scaling and encoding."""
    # Encode labels to ensure they start from 0
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    if model_type == "mlp":
        # Standardize features for MLP
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y_encoded, scaler, le
    elif model_type == "cat":
        # CatBoost can handle raw features without scaling
        return X.astype(np.float32), y_encoded, None, le
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def train_mlp_model(X_train, y_train, X_val, y_val, num_features, num_classes, args, device):
    """Train MLP model."""
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    model = MLP(num_features, num_classes).to(device)
    print(f"Model: MLP with {num_features} features -> 128 hidden -> {num_classes} classes")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("Training MLP model...")
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        # Training accuracy
        train_outputs = model(X_train_tensor)
        train_probs = torch.softmax(train_outputs, dim=1)
        train_preds = torch.argmax(train_outputs, dim=1)
        train_acc = accuracy_score(y_train, train_preds.cpu().numpy())
        
        # Validation accuracy
        val_outputs = model(X_val_tensor)
        val_probs = torch.softmax(val_outputs, dim=1)
        val_preds = torch.argmax(val_outputs, dim=1)
        val_acc = accuracy_score(y_val, val_preds.cpu().numpy())
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    return model, train_probs, val_probs

def train_catboost_model(X_train, y_train, X_val, y_val, args):
    """Train CatBoost model with hyperparameter optimization."""
    
    # Determine if this is binary or multi-class classification
    num_classes = len(np.unique(y_train))
    print(f"Detected {num_classes} classes in training data")
    
    def objective(trial):
        param = {
            "depth": trial.suggest_int("depth", 1, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1, log=True),
            "random_strength": trial.suggest_float("random_strength", 1, 10, log=True),
            "iterations": trial.suggest_int("iterations", 1, 100, log=True),
        }
        
        # Set objective based on number of classes
        if num_classes == 2:
            param["objective"] = trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"])
        else:
            param["objective"] = "MultiClass"
        
        param["thread_count"] = 4
        param["random_seed"] = args.seed
        
        _X_train, _X_valid, _y_train, _y_valid = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, 
            random_state=np.random.randint(0, 1000)
        )
        clf = CatBoostClassifier(**param)
        clf.fit(_X_train, _y_train, verbose=0)
        
        # Use accuracy for multi-class classification instead of ROC AUC
        _y_pred_test = clf.predict(_X_valid)
        score = accuracy_score(_y_valid, _y_pred_test)
        
        return score
    
    print("Training CatBoost model with hyperparameter optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5, n_jobs=30)
    
    # Train final model with best parameters
    best_params = study.best_params
    
    # Set objective for final model
    if num_classes == 2:
        # Keep the suggested objective for binary classification
        pass
    else:
        best_params["objective"] = "MultiClass"
    
    best_params["thread_count"] = 4
    best_params["random_seed"] = args.seed
    
    print(f"Best parameters: {best_params}")
    
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, verbose=0)
    
    # Get predictions
    train_probs = model.predict(X_train, prediction_type="Probability")
    val_probs = model.predict(X_val, prediction_type="Probability")
    
    # Compute accuracies
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    return model, train_probs, val_probs

def main():
    parser = argparse.ArgumentParser(description="Base model training for tabular datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["purchase", "texas"], help="Dataset name")
    parser.add_argument("--data_root", type=str, default="./data/", help="Data root directory")
    parser.add_argument("--model_root", type=str, default="./tabular/models/", help="Model save directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "cat"], help="Type of base model")
    
    # MLP-specific parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create model directory
    os.makedirs(args.model_root, exist_ok=True)
    
    # Load and preprocess data
    print(f"Loading {args.dataset} dataset...")
    X, y = load_dataset(args.dataset, args.data_root)
    X, y, scaler, label_encoder = preprocess_data(X, y, args.model_type)
    
    print(f"Dataset shape: {X.shape}, Labels: {len(np.unique(y))} classes")
    print(f"Class distribution: {np.bincount(y)[:10]}...")  # Show first 10 classes
    
    # Split data into train/test (50/50 split as in original ACS script)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )
    
    # Further split test set into val/holdout (80/20 split)
    X_val, X_holdout, y_val, y_holdout = train_test_split(
        X_test, y_test, test_size=0.2, random_state=42, stratify=y_test
    )
    
    # Create final test set (train + holdout for MIA evaluation)
    X_test_final = np.concatenate((X_train, X_holdout), axis=0)
    y_test_final = np.concatenate((y_train, y_holdout), axis=0)
    # FIXED: Correct membership assignment - train=members, holdout=nonmembers
    membership = ["in"] * len(X_train) + ["out"] * len(X_holdout)
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set (for MIA): {X_test_final.shape[0]} samples ({len(X_train)} in, {len(X_holdout)} out)")
    
    # Train model based on type
    num_classes = len(np.unique(y))
    num_features = X.shape[1]
    
    if args.model_type == "mlp":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model, train_probs_for_mia, val_probs = train_mlp_model(
            X_train, y_train, X_val, y_val, num_features, num_classes, args, device
        )
        
        # Get holdout predictions
        holdout_tensor = torch.FloatTensor(X_holdout).to(device)
        model.eval()
        with torch.no_grad():
            holdout_outputs = model(holdout_tensor)
            holdout_probs = torch.softmax(holdout_outputs, dim=1)
        
        # Convert to lists for CSV storage
        train_probs_list = [prob.cpu().numpy().tolist() for prob in train_probs_for_mia]
        val_probs_list = [prob.cpu().numpy().tolist() for prob in val_probs]
        holdout_probs_list = [prob.cpu().numpy().tolist() for prob in holdout_probs]
        
    elif args.model_type == "cat":
        model, train_probs_for_mia, val_probs = train_catboost_model(
            X_train, y_train, X_val, y_val, args
        )
        
        # Get holdout predictions
        holdout_probs = model.predict(X_holdout, prediction_type="Probability")
        
        # Convert to lists for CSV storage
        train_probs_list = [prob.tolist() for prob in train_probs_for_mia]
        val_probs_list = [prob.tolist() for prob in val_probs]
        holdout_probs_list = [prob.tolist() for prob in holdout_probs]
    
    # Save model and preprocessors with model type in path
    model_save_dir = os.path.join(args.model_root, f"{args.dataset}_{args.model_type}_base_model")
    os.makedirs(model_save_dir, exist_ok=True)
    
    if args.model_type == "mlp":
        torch.save(model.state_dict(), os.path.join(model_save_dir, "model.pth"))
        torch.save({
            'scaler': scaler,
            'label_encoder': label_encoder,
            'num_features': num_features,
            'num_classes': num_classes,
            'model_type': args.model_type
        }, os.path.join(model_save_dir, "preprocessors.pth"))
    elif args.model_type == "cat":
        # Save CatBoost model
        model.save_model(os.path.join(model_save_dir, "model.cbm"))
        torch.save({
            'scaler': scaler,  # Will be None for CatBoost
            'label_encoder': label_encoder,
            'num_features': num_features,
            'num_classes': num_classes,
            'model_type': args.model_type
        }, os.path.join(model_save_dir, "preprocessors.pth"))
    
    # Save processed features and labels for QMIA
    # Save them in the same order as the predictions CSV for proper alignment
    X_train_ordered = X_train
    y_train_ordered = y_train  
    X_val_ordered = X_val
    y_val_ordered = y_val
    X_holdout_ordered = X_holdout
    y_holdout_ordered = y_holdout
    
    # Combine in the same order as the CSV (train, val, holdout)
    X_combined_ordered = np.concatenate([X_train_ordered, X_val_ordered, X_holdout_ordered], axis=0)
    y_combined_ordered = np.concatenate([y_train_ordered, y_val_ordered, y_holdout_ordered], axis=0)
    
    torch.save({
        'X_combined': X_combined_ordered,
        'y_combined': y_combined_ordered,
        'train_size': len(X_train_ordered),
        'val_size': len(X_val_ordered),
        'holdout_size': len(X_holdout_ordered)
    }, os.path.join(model_save_dir, "processed_data.pth"))
    
    # FIXED: Proper data organization for two-mask approach
    # Train data: always members for MIA evaluation
    data_train = {
        "Score": train_probs_list,
        "Label": y_train.tolist(),
        "Membership": ["in"] * len(y_train),  # Train = members
        "Split": ["train"] * len(y_train)
    }
    
    # Val data: used for QMIA training, should be labeled as non-members (like ACS)
    data_val = {
        "Score": val_probs_list,
        "Label": y_val.tolist(),
        "Membership": ["out"] * len(y_val),  # Non-members for QMIA training (like ACS)
        "Split": ["val"] * len(y_val)
    }
    
    # Holdout data: always non-members for MIA evaluation  
    data_holdout = {
        "Score": holdout_probs_list,
        "Label": y_holdout.tolist(),
        "Membership": ["out"] * len(y_holdout),  # Holdout = non-members
        "Split": ["test"] * len(y_holdout)
    }
    
    # Create DataFrames
    df_train = pd.DataFrame(data_train)
    df_val = pd.DataFrame(data_val)
    df_holdout = pd.DataFrame(data_holdout)
    df_combined = pd.concat([df_train, df_val, df_holdout], axis=0, ignore_index=True)
    
    # Save to CSV
    predictions_path = os.path.join(model_save_dir, f"{args.dataset}_predictions.csv")
    df_combined.to_csv(predictions_path, index=False)
    
    print(f"Model saved to {model_save_dir}")
    print(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    main() 