MODEL_DIR=./models/
DATA_DIR=./data/

### Set these variables ###
BASE_ARCHITECTURE=cifar-vit
QMIA_ARCHITECTURE=facebook/convnext-tiny-224
BASE_DATASET=cifar20/0_16
ATTACK_DATASET=cifar20/0_16
###########################

# Train base model
echo "Training base model with architecture: $BASE_ARCHITECTURE" >> dropout_multisetting_tracker.log
python train_base.py --dataset=$BASE_DATASET --architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=32 --scheduler=step --scheduler_step_gamma=0.2 --scheduler_step_fraction=0.3 --lr=0.1 --weight_decay=5e-4 --epochs=200

DROPPED_CLASS_SETTINGS=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "0 1" "2 3" "4 5" "6 7" "8 9" "0 1 2 3 4" "5 6 7 8 9" "0 1 2 3 4 5 6 7 8 9" "10 11 12 13 14 15 16 17 18 19")

# Train QMIA
for DROPPED_CLASSES in "${DROPPED_CLASS_SETTINGS[@]}"; do
    echo "Training QMIA with dropped classes: ${DROPPED_CLASSES}" >> dropout_multisetting_tracker.log
    python train_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
    --batch_size=16 --image_size=224 --epochs=30 --score_fn top_two_margin --loss_fn gaussian --cls_drop ${DROPPED_CLASSES}
    # Evaluate performance
    echo "Evaluating QMIA with dropped classes: ${DROPPED_CLASSES}" >> dropout_multisetting_tracker.log
    python evaluate_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
    --batch_size=128 --image_size=224 --score_fn top_two_margin --loss_fn gaussian --checkpoint last --cls_drop ${DROPPED_CLASSES}
done