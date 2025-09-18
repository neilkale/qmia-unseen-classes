MODEL_DIR=./models/
DATA_DIR=./data/

### Set these variables ###
BASE_ARCHITECTURE=tabular-texas-mlp
QMIA_ARCHITECTURE=tabular-texas-mlp-large
BASE_DATASET=texas/0_16
ATTACK_DATASET=texas/0_16
DROPPED_CLASSES=(0 1 2 3 4 5 6 7 8 9)
###########################


# Train base model
python train_base.py --dataset=$BASE_DATASET --architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=128 --scheduler=step --scheduler_step_gamma=1 --scheduler_step_fraction=0.3 --optimizer=adamw --lr=1e-3 --weight_decay=0.0 --epochs=5 --rerun

# # Train QMIA
python train_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=16 --image_size=224 --epochs=1 --score_fn top_two_margin --loss_fn gaussian --cls_drop "${DROPPED_CLASSES[@]}"

# # Evaluate performance
python evaluate_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=128 --image_size=224 --score_fn top_two_margin --loss_fn gaussian --checkpoint last --cls_drop "${DROPPED_CLASSES[@]}"