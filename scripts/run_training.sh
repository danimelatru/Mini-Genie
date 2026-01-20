#!/bin/bash
#SBATCH --job-name=minigenie_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

# --- ENVIRONMENT SETUP ---
module purge
source ~/.bashrc

conda activate mini-genie 

# --- PATH CONFIGURATION ---
PROJECT_ROOT="/gpfs/workdir/fernandeda/mini-genie"

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p logs

echo "--- Job Started: $SLURM_JOB_ID ---"
echo "Project Root: $PROJECT_ROOT"
echo "Python Path: $(which python)"

# --- EXECUTION ---

# 2. Train the new World Model (Transformer)
echo "Step 2: Training Transformer Dynamics..."
python src/train_transformer_dynamics.py

echo "--- Job Finished ---"