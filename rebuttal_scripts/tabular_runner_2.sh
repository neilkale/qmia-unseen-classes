# Drop last 10 classes (keep first 90)
python tabular/tabular_runner.py --dataset purchase --epochs 5 --model_type cat --custom_dropped_classes "10-99"
python tabular/tabular_runner.py --dataset purchase --epochs 5 --model_type mlp --custom_dropped_classes "10-99"

# # Drop 20%, 50%, 80% of classes  
# python tabular/tabular_runner.py --dataset texas --dropped_ratios 0.2 0.5 0.8

# # Quick start (drops 20%, 50%, 80% by default)
# python tabular/tabular_runner.py --dataset texas