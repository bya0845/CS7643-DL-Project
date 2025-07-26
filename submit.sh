#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:4   # change GPU count -C HX00  
#SBATCH --mem-per-gpu=128GB    
#SBATCH -J CCT_MANUAL_PRUNING
#SBATCH -o /home/hice1/byang364/scratch/cs7643-PACE/part2-pytorch/slurm_outs/%x_%j.out

export MODEL_NAME="cct_6_3x1_32" #cct_6_3x1_32 densenet121

mkdir -p /home/hice1/byang364/scratch/cs7643-PACE/part2-pytorch/slurm_outs
module load anaconda3/2023.03

NUM_GPUS=4 # change if # of GPU changes

echo "Setting up environment paths..."
export CONDA_ENV_PATH="/home/hice1/byang364/scratch/cs7643-PACE"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export PYTHONPATH="$CONDA_ENV_PATH/lib/python3.12/site-packages:$PYTHONPATH"

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # for 8 GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 # for 6 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3 # for 4 GPUs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd /home/hice1/byang364/scratch/cs7643-PACE/part2-pytorch/

accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --machine_rank 0 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --main_process_port 0 \
    train_model.py 2>&1 | grep -v "ipex flag is deprecated"

echo "Training finished at $(date)"