#!/bin/bash
# Train UCOD-DPL (baseline teacher-student, no APM) and evaluate on all 4 test datasets
# The eval script automatically runs on: CHAMELEON, TE-CAMO, TE-COD10K, NC4K
#
# Usage: bash scripts/train_all_datasets.sh [-c config] [-g NUM_GPUS]
# Example: bash scripts/train_all_datasets.sh -g 1

CONFIG_FILE="./configs/uscod/UCOD-DPL_dinov2.py"
MASTER_ADDR='localhost'
MASTER_PORT=11145
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

while getopts "c:p:g:" opt; do
  case "$opt" in
    c) CONFIG_FILE="$OPTARG" ;;  
    p) MASTER_PORT="$OPTARG" ;;  
    g) GPUS_PER_NODE="$OPTARG" ;;  
    ?) echo "Usage: $0 [-c config_file] [-p master_port] [-g gpus]"; exit 1 ;;
  esac
done

DISTRIBUTED_ARGS="--mixed_precision fp16 \
                  --machine_rank $NODE_RANK\
                  --num_machines $NNODES\
                  --main_process_port $MASTER_PORT \
                  --num_processes $GPUS_PER_NODE"

if [ $GPUS_PER_NODE -gt 1 ]; then
  DISTRIBUTED_ARGS="$DISTRIBUTED_ARGS --multi_gpu"
fi

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=./
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "============================================"
echo "  UCOD-DPL Baseline Training (No APM)"
echo "  Config: $CONFIG_FILE"
echo "  GPUs: $GPUS_PER_NODE"
echo "  Train set: TR-CAMO + TR-COD10K + CHAMELEON + NC4K + TE-CAMO + TE-COD10K"
echo "  Eval sets: CHAMELEON, TE-CAMO, TE-COD10K, NC4K"
echo "============================================"

echo ""
echo "[STEP 1] Generating Pseudo Labels (Cache)..."
echo "This might take 10-15 minutes if not already cached."
PYTHONPATH=./ python scripts/../generate_pseudo_label.py --dataset TR-CAMO+TR-COD10K+CHAMELEON+NC4K+TE-CAMO+TE-COD10K

# Train
TRAIN_CMD="accelerate launch $DISTRIBUTED_ARGS scripts/train.py --config $CONFIG_FILE"
echo ""
echo "[STEP 2] Training..."
echo "$TRAIN_CMD"
${TRAIN_CMD}

echo ""
echo "============================================"
echo "  Training Complete!"
echo "============================================"

echo ""
echo "[STEP 3] Evaluating on all 4 Test Datasets..."
# We use the final epoch's checkpoint for evaluation.
# Assuming max_epoch is 25 (from config), so epoch25.pth
CHECKPOINT_DIR="run/uscod/UCOD-DPL_dinov2/ckp"
LATEST_CKPT=$(ls -t ${CHECKPOINT_DIR}/epoch*.pth 2>/dev/null | head -1)

if [ -f "$LATEST_CKPT" ]; then
    echo "Found checkpoint: $LATEST_CKPT"
    bash scripts/launch_val_first_stage.sh -c $CONFIG_FILE -m $LATEST_CKPT -g $GPUS_PER_NODE
else
    echo "No checkpoint found in ${CHECKPOINT_DIR} to evaluate!"
fi
