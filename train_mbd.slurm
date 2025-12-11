#!/bin/bash
#SBATCH --job-name=mbd_train
#SBATCH --output=logs/train_mbd_%j.out
#SBATCH --error=logs/train_mbd_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================================================
# Monotone-by-Design (MbD) Model Training for Battery RUL Prediction
# ============================================================================
# This script trains a PyTorch MbD model that guarantees monotonicity:
#   as cycle increases, predicted RUL does NOT increase (global guarantee)
# ============================================================================

# Print job information
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Setup environment
module load miniconda

# Activate conda environment (try both methods for compatibility)
if command -v conda &> /dev/null; then
    conda activate nnv || source activate nnv
else
    source activate nnv
fi

# Change to script directory
cd /gpfs/radev/scratch/xu_hua/mj756/course_proj/NNV

# Print environment info
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  Working directory: $(pwd)"
echo "  Conda env: $CONDA_DEFAULT_ENV"
CUDA_AVAIL=$(python -c 'import torch; print(torch.cuda.is_available())')
echo "  CUDA available: $CUDA_AVAIL"
if [ "$CUDA_AVAIL" = "True" ]; then
    echo "  GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Set default paths and parameters (can be overridden via environment variables)
DATA_DIR="${DATA_DIR:-./prep_out}"
OUT_DIR="${OUT_DIR:-./mbd_pt}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.0015}"

# Print configuration
echo "============================================================"
echo "Training Configuration"
echo "============================================================"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo ""

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory '$DATA_DIR' does not exist!"
    echo "Please run preprocess.py first to generate the data."
    exit 1
fi

if [ ! -f "$DATA_DIR/metadata.json" ]; then
    echo "ERROR: metadata.json not found in '$DATA_DIR'!"
    echo "Please run preprocess.py first to generate the data."
    exit 1
fi

# Run training
echo "============================================================"
echo "Starting MbD Model Training"
echo "============================================================"
echo ""

python mbd_train.py \
    --data_dir "$DATA_DIR" \
    --out_dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR"

# Check exit status
EXIT_CODE=$?
echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "  Output directory: $OUT_DIR"
    echo "  Metrics saved to: $OUT_DIR/metrics.json"
    echo "  ONNX model saved to: $OUT_DIR/model.onnx"
else
    echo "Training failed with exit code: $EXIT_CODE"
    echo "Check the error log for details: logs/train_mbd_${SLURM_JOB_ID}.err"
fi
echo "End Time: $(date)"
echo "============================================================"

exit $EXIT_CODE

