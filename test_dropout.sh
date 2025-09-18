MODEL_DIR=./models/
DATA_DIR=./data/

### Set these variables ###
BASE_ARCHITECTURE=cifar-resnet-50
QMIA_ARCHITECTURE=facebook/convnext-tiny-224 
BASE_DATASET=cifar20/0_16
ATTACK_DATASET=cifar20/0_16
DROPPED_CLASSES=(0)
###########################

# Train base model
python train_base.py --dataset=$BASE_DATASET --architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=32 --scheduler=step --scheduler_step_gamma=0.2 --scheduler_step_fraction=0.3 --lr=0.1 --weight_decay=5e-4 --epochs=100 --rerun

# Train QMIA
python train_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=16 --image_size=224 --epochs=30 --score_fn top_two_margin --loss_fn gaussian --cls_drop "${DROPPED_CLASSES[@]}"

# Evaluate performance
python evaluate_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=128 --image_size=224 --score_fn top_two_margin --loss_fn gaussian --checkpoint last --cls_drop "${DROPPED_CLASSES[@]}"