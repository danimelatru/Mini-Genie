#!/bin/bash
#SBATCH --job-name=gen_gif
#SBATCH --output=logs/viz_%j.out
#SBATCH --time=00:10:00
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

module purge
PROJECT_ROOT="/gpfs/workdir/fernandeda/mini-genie"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
PYTHON_EXEC="/gpfs/workdir/fernandeda/conda_envs/mini-genie/bin/python"

$PYTHON_EXEC src/generate_dream_gif.py