python tabular/tabular_qmia_training.py --dataset purchase --model_type mlp --dropped_classes 90 91 92 93 94 95 96 97 98 99 --qmia_model mlp --score_function top_two_margin

# Drop last 10 classes (keep first 90)
# python tabular/tabular_runner.py --dataset purchase --epochs 5 --model_type cat --custom_dropped_classes "1-99"
# python tabular/tabular_runner.py --dataset purchase --epochs 5 --model_type mlp --custom_dropped_classes "1-99"

# # Drop 20%, 50%, 80% of classes  
# python tabular/tabular_runner.py --dataset texas --dropped_ratios 0.2 0.5 0.8

# # Quick start (drops 20%, 50%, 80% by default)
# python tabular/tabular_runner.py --dataset texas