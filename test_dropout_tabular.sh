MODEL_DIR=./models/
DATA_DIR=./data/

### Set these variables ###
BASE_ARCHITECTURE=mlp-texas-tiny
QMIA_ARCHITECTURE=mlp-texas-small
BASE_DATASET=texas/0_16
ATTACK_DATASET=texas/0_16
DROPPED_CLASSES_SETTINGS=("20-30" "30-40" "40-50" "50-60" "60-70" "70-80" "80-90" "90-100") # Examples: ("0-10"), (0,1,2,3,4,5,6,7,8,9)
###########################

# Train base model
python train_base.py --dataset=$BASE_DATASET --architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=32 --scheduler=step --scheduler_step_gamma=1 --scheduler_step_fraction=0.3 --lr=1e-3 --optimizer=adamw --weight_decay=5e-4 --epochs=30

for DROPPED_CLASSES in "${DROPPED_CLASSES_SETTINGS[@]}"; do
    # Train QMIA
    python train_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
    --batch_size=16 --image_size=224 --epochs=20 --score_fn top_two_margin --loss_fn gaussian --cls_drop_range "${DROPPED_CLASSES[@]}"
    # Evaluate performance
    python evaluate_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
    --batch_size=128 --image_size=224 --score_fn top_two_margin --loss_fn gaussian --checkpoint last --cls_drop_range "${DROPPED_CLASSES[@]}" --rerun
done