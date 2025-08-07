# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_summaries(logs_root, dataset, model_type="mlp", seed=None):
    """Load all summary files for a given dataset and model type."""
    if seed is not None:
        pattern = os.path.join(logs_root, f"{dataset}_{model_type}_qmia_summary_*_seed_{seed}.csv")
    else:
        pattern = os.path.join(logs_root, f"{dataset}_{model_type}_qmia_summary_*.csv")
    
    summary_files = glob.glob(pattern)
    
    if not summary_files:
        print(f"No summary files found for {dataset} ({model_type}) in {logs_root}")
        return None
    
    print(f"Found {len(summary_files)} summary files")
    
    all_summaries = []
    for file in summary_files:
        try:
            df = pd.read_csv(file)
            all_summaries.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if all_summaries:
        return pd.concat(all_summaries, ignore_index=True)
    else:
        return None

def compute_summary_stats(df):
    """Compute summary statistics across experiments."""
    
    # Extract seen class info - handle both old (seen_classes) and new (dropped_classes) formats
    if 'dropped_classes' in df.columns:
        # New format: calculate seen classes from dropped classes
        df['num_dropped_classes'] = df['dropped_classes'].apply(lambda x: len(eval(x)) if isinstance(x, str) else len(x))
        df['num_seen_classes'] = 100 - df['num_dropped_classes']  # Assuming 100 total classes
    else:
        # Old format: use seen_classes directly  
        df['num_seen_classes'] = df['seen_classes'].apply(lambda x: len(eval(x)) if isinstance(x, str) else len(x))
    
    df['seen_ratio'] = df['num_seen_classes'] / 100  # Assuming 100 total classes
    
    # Group by seen ratio
    grouped = df.groupby('seen_ratio').agg({
        'overall_qmia_auc': ['mean', 'std'],
        'overall_baseline_auc': ['mean', 'std'],
        'overall_qmia_tpr_at_1_fpr': ['mean', 'std'],
        'overall_baseline_tpr_at_1_fpr': ['mean', 'std'],
        'seen_qmia_auc': ['mean', 'std'],
        'seen_baseline_auc': ['mean', 'std'],
        'seen_qmia_tpr_at_1_fpr': ['mean', 'std'],
        'seen_baseline_tpr_at_1_fpr': ['mean', 'std'],
        'unseen_qmia_auc': ['mean', 'std'],
        'unseen_baseline_auc': ['mean', 'std'],
        'unseen_qmia_tpr_at_1_fpr': ['mean', 'std'],
        'unseen_baseline_tpr_at_1_fpr': ['mean', 'std']
    }).round(4)
    
    return grouped

def plot_results(df, dataset, model_type, save_path=None):
    """Create visualization of results."""
    
    # Extract seen class info - handle both old and new formats
    if 'dropped_classes' in df.columns:
        df['num_dropped_classes'] = df['dropped_classes'].apply(lambda x: len(eval(x)) if isinstance(x, str) else len(x))
        df['num_seen_classes'] = 100 - df['num_dropped_classes']
    else:
        df['num_seen_classes'] = df['seen_classes'].apply(lambda x: len(eval(x)) if isinstance(x, str) else len(x))
    
    df['seen_ratio'] = df['num_seen_classes'] / 100
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'QMIA Performance vs Class Dropout - {dataset.upper()} ({model_type.upper()})', fontsize=16)
    
    # Plot 1: AUC comparison
    ax1 = axes[0, 0]
    seen_ratios = sorted(df['seen_ratio'].unique())
    
    overall_qmia_aucs = [df[df['seen_ratio'] == r]['overall_qmia_auc'].mean() for r in seen_ratios]
    overall_baseline_aucs = [df[df['seen_ratio'] == r]['overall_baseline_auc'].mean() for r in seen_ratios]
    
    ax1.plot(seen_ratios, overall_qmia_aucs, 'o-', label='QMIA', linewidth=2, markersize=8)
    ax1.plot(seen_ratios, overall_baseline_aucs, 's-', label='Baseline', linewidth=2, markersize=8)
    ax1.set_xlabel('Fraction of Classes Seen During QMIA Training')
    ax1.set_ylabel('Overall AUC')
    ax1.set_title('Overall Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: TPR at 1% FPR comparison
    ax2 = axes[0, 1]
    overall_qmia_tprs = [df[df['seen_ratio'] == r]['overall_qmia_tpr_at_1_fpr'].mean() for r in seen_ratios]
    overall_baseline_tprs = [df[df['seen_ratio'] == r]['overall_baseline_tpr_at_1_fpr'].mean() for r in seen_ratios]
    
    ax2.plot(seen_ratios, overall_qmia_tprs, 'o-', label='QMIA', linewidth=2, markersize=8)
    ax2.plot(seen_ratios, overall_baseline_tprs, 's-', label='Baseline', linewidth=2, markersize=8)
    ax2.set_xlabel('Fraction of Classes Seen During QMIA Training')
    ax2.set_ylabel('Overall TPR at 1% FPR')
    ax2.set_title('Overall TPR at 1% FPR')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Seen vs Unseen AUC
    ax3 = axes[1, 0]
    seen_qmia_aucs = [df[df['seen_ratio'] == r]['seen_qmia_auc'].mean() for r in seen_ratios]
    unseen_qmia_aucs = [df[df['seen_ratio'] == r]['unseen_qmia_auc'].mean() for r in seen_ratios]
    
    ax3.plot(seen_ratios, seen_qmia_aucs, 'o-', label='Seen Classes (QMIA)', linewidth=2, markersize=8)
    ax3.plot(seen_ratios, unseen_qmia_aucs, 's-', label='Unseen Classes (QMIA)', linewidth=2, markersize=8)
    ax3.set_xlabel('Fraction of Classes Seen During QMIA Training')
    ax3.set_ylabel('AUC')
    ax3.set_title('Seen vs Unseen Classes Performance (AUC)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Seen vs Unseen TPR at 1% FPR
    ax4 = axes[1, 1]
    seen_qmia_tprs = [df[df['seen_ratio'] == r]['seen_qmia_tpr_at_1_fpr'].mean() for r in seen_ratios]
    unseen_qmia_tprs = [df[df['seen_ratio'] == r]['unseen_qmia_tpr_at_1_fpr'].mean() for r in seen_ratios]
    
    ax4.plot(seen_ratios, seen_qmia_tprs, 'o-', label='Seen Classes (QMIA)', linewidth=2, markersize=8)
    ax4.plot(seen_ratios, unseen_qmia_tprs, 's-', label='Unseen Classes (QMIA)', linewidth=2, markersize=8)
    ax4.set_xlabel('Fraction of Classes Seen During QMIA Training')
    ax4.set_ylabel('TPR at 1% FPR')
    ax4.set_title('Seen vs Unseen Classes Performance (TPR at 1% FPR)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def create_results_table(df):
    """Create a nicely formatted results table."""
    
    # Extract seen class info - handle both old and new formats
    if 'dropped_classes' in df.columns:
        df['num_dropped_classes'] = df['dropped_classes'].apply(lambda x: len(eval(x)) if isinstance(x, str) else len(x))
        df['num_seen_classes'] = 100 - df['num_dropped_classes']
    else:
        df['num_seen_classes'] = df['seen_classes'].apply(lambda x: len(eval(x)) if isinstance(x, str) else len(x))
    
    df['seen_ratio'] = df['num_seen_classes'] / 100
    
    # Create summary table
    table_data = []
    
    for ratio in sorted(df['seen_ratio'].unique()):
        subset = df[df['seen_ratio'] == ratio]
        
        row = {
            'Seen Ratio': f"{ratio:.1%}",
            'Num Seen Classes': int(ratio * 100),
            'Overall QMIA AUC': f"{subset['overall_qmia_auc'].mean():.3f} ± {subset['overall_qmia_auc'].std():.3f}",
            'Overall Baseline AUC': f"{subset['overall_baseline_auc'].mean():.3f} ± {subset['overall_baseline_auc'].std():.3f}",
            'Seen QMIA AUC': f"{subset['seen_qmia_auc'].mean():.3f} ± {subset['seen_qmia_auc'].std():.3f}",
            'Unseen QMIA AUC': f"{subset['unseen_qmia_auc'].mean():.3f} ± {subset['unseen_qmia_auc'].std():.3f}",
            'Seen TPR@1%': f"{subset['seen_qmia_tpr_at_1_fpr'].mean():.3f} ± {subset['seen_qmia_tpr_at_1_fpr'].std():.3f}",
            'Unseen TPR@1%': f"{subset['unseen_qmia_tpr_at_1_fpr'].mean():.3f} ± {subset['unseen_qmia_tpr_at_1_fpr'].std():.3f}"
        }
        table_data.append(row)
    
    results_table = pd.DataFrame(table_data)
    return results_table

def main():
    parser = argparse.ArgumentParser(description="Evaluate tabular QMIA experiment results")
    parser.add_argument("--dataset", type=str, required=True, choices=["purchase", "texas"], 
                       help="Dataset name")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "cat"], 
                       help="Type of base model")
    parser.add_argument("--logs_root", type=str, default="./tabular/logs/", 
                       help="Logs directory")
    parser.add_argument("--seed", type=int, default=None, 
                       help="Specific seed to analyze (if None, analyze all seeds)")
    parser.add_argument("--save_plots", action="store_true", 
                       help="Save plots to files")
    parser.add_argument("--save_table", action="store_true", 
                       help="Save results table to CSV")
    
    args = parser.parse_args()
    
    print(f"Analyzing results for {args.dataset} dataset with {args.model_type} base model")
    print(f"Logs directory: {args.logs_root}")
    
    # Load all summary data
    df = load_all_summaries(args.logs_root, args.dataset, args.model_type, args.seed)
    
    if df is None or len(df) == 0:
        print("No data found. Make sure you have run the experiments first.")
        return
    
    print(f"Loaded {len(df)} experiment results")
    
    # Display basic info
    print(f"\nDataset: {args.dataset}")
    print(f"Model type: {args.model_type}")
    print(f"Number of experiments: {len(df)}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    
    seen_classes_info = []
    dropped_classes_info = []
    for _, row in df.iterrows():
        if 'dropped_classes' in df.columns:
            dropped_classes = eval(row['dropped_classes']) if isinstance(row['dropped_classes'], str) else row['dropped_classes']
            dropped_classes_info.append(len(dropped_classes))
            seen_classes_info.append(100 - len(dropped_classes))  # Assuming 100 total classes
        else:
            # Old format
            seen_classes = eval(row['seen_classes']) if isinstance(row['seen_classes'], str) else row['seen_classes']
            seen_classes_info.append(len(seen_classes))
    
    print(f"Seen class counts: {sorted(set(seen_classes_info))}")
    if dropped_classes_info:
        print(f"Dropped class counts: {sorted(set(dropped_classes_info))}")
    
    # Compute summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    summary_stats = compute_summary_stats(df)
    print(summary_stats)
    
    # Create results table
    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    
    results_table = create_results_table(df)
    print(results_table.to_string(index=False))
    
    if args.save_table:
        table_path = os.path.join(args.logs_root, f"{args.dataset}_{args.model_type}_results_table.csv")
        results_table.to_csv(table_path, index=False)
        print(f"\nResults table saved to {table_path}")
    
    # Create plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    if args.save_plots:
        plot_path = os.path.join(args.logs_root, f"{args.dataset}_{args.model_type}_results_plot.png")
        plot_results(df, args.dataset, args.model_type, save_path=plot_path)
    else:
        plot_results(df, args.dataset, args.model_type)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 