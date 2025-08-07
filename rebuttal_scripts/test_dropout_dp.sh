MODEL_DIR=./models/
DATA_DIR=./data/

### Set these variables ###
BASE_ARCHITECTURE=cifar-resnet-50-dp
QMIA_ARCHITECTURE=facebook/convnext-tiny-224 
BASE_DATASET=cifar20/0_16
ATTACK_DATASET=cifar20/0_16
DROPPED_CLASSES=(0)

### Differential Privacy settings ###
### Using RDP accountant (no more PRV division by zero issues!) ###

DP_ENABLE="--enable_dp"
DP_MAX_GRAD_NORM="--dp_max_grad_norm 1.0"
DP_TARGET_EPSILON="--dp_target_epsilon 8.0"
DP_TARGET_DELTA="--dp_target_delta 8e-5"
DP_NOISE_MULTIPLIER="--dp_noise_multiplier 1.0"
# DP_SECURE_RNG="--dp_secure_rng"  # Uncomment for production

# Train base model with DP
python train_base.py --dataset=$BASE_DATASET --architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=32 --scheduler=step --scheduler_step_gamma=0.2 --scheduler_step_fraction=0.3 --lr=0.1 --weight_decay=5e-4 --epochs=10 \
$DP_ENABLE $DP_NOISE_MULTIPLIER $DP_MAX_GRAD_NORM $DP_TARGET_EPSILON $DP_TARGET_DELTA $DP_SECURE_RNG

# Train QMIA (no DP needed for attack model)
python train_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=16 --image_size=224 --epochs=3 --score_fn top_two_margin --loss_fn gaussian --cls_drop "${DROPPED_CLASSES[@]}" \
$DP_ENABLE $DP_MAX_GRAD_NORM $DP_TARGET_EPSILON $DP_TARGET_DELTA $DP_SECURE_RNG

# Evaluate performance
python evaluate_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=128 --image_size=224 --score_fn top_two_margin --loss_fn gaussian --checkpoint last --cls_drop "${DROPPED_CLASSES[@]}" \
$DP_ENABLE $DP_MAX_GRAD_NORM $DP_TARGET_EPSILON $DP_TARGET_DELTA $DP_SECURE_RNG