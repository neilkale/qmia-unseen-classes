MODEL_DIR=./models/
DATA_DIR=./data/

## Fixed DP parameters ##
DP_ENABLE="--enable_dp"
DP_MAX_GRAD_NORM="--dp_max_grad_norm 3.0"
DP_TARGET_EPSILON="--dp_target_epsilon 50.0"
DP_TARGET_DELTA="--dp_target_delta 1e-6"
DP_NOISE_MULTIPLIER="--dp_noise_multiplier -1"
###########################

### Set these variables ###
BASE_ARCHITECTURE=cifar-resnet-10-dp
QMIA_ARCHITECTURE=facebook/convnext-tiny-224 
BASE_DATASET=cifar20/0_16
ATTACK_DATASET=cifar20/0_16
###########################

export CUDA_VISIBLE_DEVICES=6

# Optional secure RNG flag for DP (uncomment to enable)
# DP_SECURE_RNG="--dp_secure_rng"
DP_SECURE_RNG=""

# Train base model
echo "Training base model with architecture: $BASE_ARCHITECTURE" >> dropout_multisetting_tracker.log
python train_base.py --dataset=$BASE_DATASET --architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=64 --lr=1e-4 --weight_decay=5e-4 --epochs=40 --optimizer=adamw --rerun \
$DP_ENABLE $DP_NOISE_MULTIPLIER $DP_MAX_GRAD_NORM $DP_TARGET_EPSILON $DP_TARGET_DELTA $DP_SECURE_RNG 
