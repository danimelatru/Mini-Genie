#!/bin/bash
#SBATCH --job-name=minigenie_full
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

# --- ENVIRONMENT SETUP ---
module purge
# If your cluster requires loading specific CUDA modules, uncomment below:
# module load cuda/11.8

# --- PATH CONFIGURATION ---
PROJECT_ROOT="/gpfs/workdir/fernandeda/mini-genie"
cd "$PROJECT_ROOT"

# Critical: Add project root to PYTHONPATH so 'src' imports work
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create logs directory to prevent startup errors
mkdir -p logs

# --- PYTHON EXECUTABLE ---
# We use the absolute path to ensure PyTorch is found (avoids 'conda activate' issues)
PYTHON_EXEC="/gpfs/workdir/fernandeda/conda_envs/mini-genie/bin/python"

echo "--- Job Started: $SLURM_JOB_ID ---"
echo "Project Root: $PROJECT_ROOT"
echo "Using Python: $PYTHON_EXEC"

# --- EXECUTION ---

# 1. Tokenize Data 
# (Keep commented unless you changed VQ-VAE or need to regenerate tokens)
# echo "Step 1: Tokenizing Data..."
# $PYTHON_EXEC src/tokenize_data.py

# 2. Train the World Model (Transformer)
echo "Step 2: Training Transformer Dynamics..."
$PYTHON_EXEC src/train_transformer_dynamics.py

# 3. Generate Dream GIF (Visualization)
# This creates 'data/artifacts/dream_sequence.gif'
echo "Step 3: Generating Dream Visualization..."
$PYTHON_EXEC src/generate_dream_gif.py

echo "--- Job Finished ---"