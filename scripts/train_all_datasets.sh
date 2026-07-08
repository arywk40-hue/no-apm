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
echo "  Train set: TR-CAMO + TR-COD10K"
echo "  Eval sets: CHAMELEON, TE-CAMO, TE-COD10K, NC4K"
echo "============================================"

# Train
TRAIN_CMD="accelerate launch $DISTRIBUTED_ARGS scripts/train.py --config $CONFIG_FILE"
echo ""
echo "[STEP 1] Training..."
echo "$TRAIN_CMD"
${TRAIN_CMD}

echo ""
echo "============================================"
echo "  Training Complete!"
echo "============================================"
