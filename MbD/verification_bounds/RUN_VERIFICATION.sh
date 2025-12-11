#!/bin/bash
# Script to run α-β-CROWN verification with correct paths

# Activate conda environment
conda activate /gpfs/radev/scratch/xu_hua/mj756/conda_envs/alpha-beta-crown

# Navigate to α-β-CROWN directory
cd /gpfs/radev/scratch/xu_hua/mj756/course_proj/NNV/alpha-beta-CROWN/complete_verifier

# Run verification with absolute path
python abcrown.py --config /gpfs/radev/scratch/xu_hua/mj756/course_proj/NNV/verification_bounds/config.yaml

