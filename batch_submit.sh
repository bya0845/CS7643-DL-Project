#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:4 -C HX00 # change GPU count 
#SBATCH --mem-per-gpu=128GB  
#SBATCH -J job
#SBATCH -o /home/hice1/byang364/scratch/cs7643-PACE/part2-pytorch/slurm_outs/%x_%j_%a.out
#SBATCH --array=0-19

export MODEL_NAME="cct_6_3x1_32"

CONFIGS=(
    "6,256,2.0,4"     # Baseline configuration
    "6,256,1.5,4"     # Reduced MLP
    "6,256,1.0,4"     # Minimal MLP
    
    # Medium models (embedding_dim=128)
    "6,128,2.0,4"     # Full layers, medium embedding
    "6,128,1.5,4"     # Reduced MLP
    "6,128,1.0,4"     # Minimal MLP
    "4,128,2.0,4"     # Fewer layers
    "4,128,1.5,4"     # Fewer layers + reduced MLP
    
    # Small models (embedding_dim=96)
    "6,96,2.0,3"      # Full layers, small embedding
    "6,96,1.5,3"      # Reduced MLP
    "4,96,2.0,3"      # Fewer layers
    "4,96,1.5,3"      # Fewer layers + reduced MLP
    "3,96,1.0,3"      # Very compact
    
    # Tiny models (embedding_dim=64)
    "6,64,2.0,2"      # Full layers, tiny embedding
    "4,64,2.0,2"      # Fewer layers
    "4,64,1.5,2"      # Fewer layers + reduced MLP
    "3,64,1.5,2"      # Very small
    "2,64,1.0,2"      # Minimal layers
    
    # Ultra-tiny models
    "3,48,1.0,3"      # Very demure
    "2,32,1.0,2"      # Very mindful
)

CURRENT_CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
IFS=',' read -r NUM_LAYERS EMBEDDING_DIM MLP_RATIO NUM_HEADS <<< "$CURRENT_CONFIG"

mkdir -p /home/hice1/byang364/scratch/cs7643-PACE/part2-pytorch/slurm_outs
module load anaconda3/2023.03

NUM_GPUS=4 # change if # of GPU changes

echo "Setting up environment paths..."
export CONDA_ENV_PATH="/home/hice1/byang364/scratch/cs7643-PACE"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export PYTHONPATH="$CONDA_ENV_PATH/lib/python3.12/site-packages:$PYTHONPATH"

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # for 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3 # for 4 GPUs

cd /home/hice1/byang364/scratch/cs7643-PACE/src

accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --machine_rank 0 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --main_process_port 0 \
    batch_train.py \
    -num_layers $NUM_LAYERS \
    -embedding_dim $EMBEDDING_DIM \
    -mlp_ratio $MLP_RATIO \
    -num_heads $NUM_HEADS

echo "Training finished at $(date)"