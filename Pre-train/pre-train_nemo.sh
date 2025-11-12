#!/bin/bash
#Author: Jingmei Yang (jmyang@bu.edu)

# ============================================================================
# SLURM BATCH CONFIGURATION
# Configure SLURM job parameters for distributed GPU training
# ============================================================================
#SBATCH --account=Your Account                    # TODO: Update account name for your system
#SBATCH --job-name=8B_NeMo_   # Job name for identification in queue
#SBATCH --error=%x_%j.err                   # Error log file (job_name_job_id.err)
#SBATCH --output=%x_%j.out                  # Output log file (job_name_job_id.out)
#SBATCH --exclusive                         # Request exclusive access to nodes
#SBATCH --gpus-per-node=8                   # Number of GPUs per compute node
#SBATCH --mem=0                             # Use all available memory on nodes
#SBATCH --nodes=5                           # Total number of compute nodes (5 nodes × 8 GPUs = 40 GPUs total)
#SBATCH --ntasks-per-node=8                 # Number of tasks per node (should match GPUs)
#SBATCH --time=10:00:00                    # Maximum job runtime 

echo "========================STARTING 8B Training with Checkpointing============================"

# ============================================================================
# JOB TIMING AND LOGGING
# Track job execution time for performance monitoring
# ============================================================================
START_TIME=$(date +%s)
echo "Job started at: $(date)"

# Load SLURM module (may vary by system)
module load slurm                           # TODO: Check if this module exists on your system

# ============================================================================
# MODEL CONFIGURATION
# Define the base model to be fine-tuned
# ============================================================================
export MODEL="Llama-3.1-8B"                # Base model identifier
echo "MODEL: $MODEL"

# ============================================================================
# NCCL (NVIDIA Collective Communication Library) CONFIGURATION
# Optimizes multi-GPU communication for distributed training
# These settings are crucial for stable training across multiple nodes
# ============================================================================
export TORCH_NCCL_AVOID_RECORD_STREAMS=1   # Avoid CUDA stream recording for memory efficiency
export NCCL_NVLS_ENABLE=0                  # Disable NVLS (may cause issues on some systems)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # Enable async error handling for better debugging
export NCCL_TIMEOUT=1800                   # Communication timeout (30 minutes)
export NCCL_IB_TIMEOUT=50                  # InfiniBand timeout settings
export NCCL_IB_RETRY_CNT=10                # Number of InfiniBand retry attempts
export NCCL_SOCKET_IFNAME=^docker0,lo      # Exclude certain network interfaces
export NCCL_IB_GID_INDEX=3                 # InfiniBand Global ID index
export NCCL_NET_GDR_LEVEL=5                # GPU Direct RDMA level
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000   # Trace buffer size for debugging

# ============================================================================
# CONTAINER AND STORAGE CONFIGURATION
# Set up paths for container image and shared storage
# ============================================================================
export SHARED_STORAGE_ROOT="/Your Path"  # TODO: Update to your shared storage path
export CONTAINER_IMAGE=$SHARED_STORAGE_ROOT/nvidia+nemo+25.04.01.sqsh  # TODO: Update container image path
export CONTAINER_WORKSPACE_MOUNT="$SHARED_STORAGE_ROOT/Your Project"  # TODO: Update workspace mount path

# ============================================================================
# AUTHENTICATION TOKENS
# Required for Weights & Biases logging and Hugging Face model access
# ============================================================================
export WANDB_API_KEY="Add your WANDB_API_KEY"           # TODO: Replace with your actual WANDB API key
export HUGGINGFACE_TOKEN="Add your HUGGINGFACE_TOKEN"   # TODO: Replace with your actual HF token
export HUGGINGFACE_HUB_TOKEN="Add your HUGGINGFACE_TOKEN"  # TODO: Replace with your actual HF token

# Set up Hugging Face cache directory
export HF_HOME="$CONTAINER_WORKSPACE_MOUNT/HF_HOME"     # TODO: Update if using different workspace path
mkdir -p "$HF_HOME"
chmod 755 "$HF_HOME"

# ============================================================================
# DISTRIBUTED TRAINING SETUP
# Configure master node and communication ports for multi-node training
# ============================================================================
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # Get first node as master
echo "MASTER_ADDR: $MASTER_ADDR"
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))  # Random port between 30000-50000
echo "MASTER_PORT: $MASTER_PORT"
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE              # Extract from SLURM environment
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
export NNODES=$SLURM_NNODES                            # Extract from SLURM environment
echo "NNODES: $NNODES"
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE) # Total number of processes (40 in this case)
echo "NUM_PROCESSES: $NUM_PROCESSES"

# ============================================================================
# DATASET AND OUTPUT CONFIGURATION
# Define input dataset and output directory for trained model
# ============================================================================
export WANDB_PROJECT="$MODEL"                          # Weights & Biases project name
export DATASET="Your Data"                      # TODO: Update dataset name if different
export DATA_PATH="$DATASET"                            # TODO: Update if dataset is in different location
export output_dir="output/NeMo/Your Project/$MODEL"  # TODO: Update output directory structure

# ============================================================================
# OUTPUT DIRECTORY CLEANUP
# Remove any existing outputs to ensure clean training start
# ============================================================================
echo "========================Cleaning Previous Outputs============================"
if [ -d "$output_dir" ]; then
    echo "Removing existing output directory: $output_dir"
    rm -rf "$output_dir"
fi
mkdir -p "$output_dir"
export PYTHONUNBUFFERED=1                              # Ensure immediate output flushing

# ============================================================================
# NEMO AUTOMODEL TRAINING PARAMETERS
# Core hyperparameters for the fine-tuning process
# ============================================================================
export MODEL_NAME="meta-llama/$MODEL"                  # Full model path for Hugging Face
export MAX_EPOCHS=1                                    # Number of training epochs
export LEARNING_RATE=2e-5                             # Learning rate (typical for LLM fine-tuning)
export GLOBAL_BATCH_SIZE=80                           # Total batch size across all GPUs
export MICRO_BATCH_SIZE=2                             # Batch size per GPU (80 total / 40 GPUs = 2)
export PRECISION="bf16"                                # Use bfloat16 for memory efficiency
export STRATEGY="fsdp2"                                # Fully Sharded Data Parallel v2 strategy
export GRADIENT_CLIP_VAL=1.0                          # Gradient clipping to prevent exploding gradients
export SEED=123                                        # Random seed for reproducibility

# ============================================================================
# CHECKPOINT CONFIGURATION
# Control model saving frequency and retention
# ============================================================================
export SAVE_EVERY_N_STEPS=1000                        # Save checkpoint every 1000 training steps
export SAVE_TOP_K=20                                  # Keep only the 20 best checkpoints
export RESUME_FROM_CHECKPOINT=false                   # Start fresh training (not resuming)

# ============================================================================
# ADDITIONAL TRAINING ARGUMENTS (HF-STYLE)
# Advanced training configuration following Hugging Face patterns
# ============================================================================
export WEIGHT_DECAY=0.0                               # L2 regularization strength
export WARMUP_RATIO=0.03                              # Fraction of training for learning rate warmup
export LR_SCHEDULER_TYPE="cosine"                     # Learning rate schedule type
export EVALUATION_STRATEGY="epoch"                    # When to run evaluation (per epoch)
export SAVE_STRATEGY="steps"                          # Save based on steps (not epochs)
export SAVE_STEPS=500                                 # Save checkpoint every 500 steps

# ============================================================================
# CONFIGURATION SUMMARY
# Display all key parameters for verification before training starts
# ============================================================================
echo "========================Configuration Summary============================"
echo "Model: $MODEL_NAME"
echo "Output Directory: $output_dir"
echo "Learning Rate: $LEARNING_RATE"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Warmup Ratio: $WARMUP_RATIO"
echo "LR Scheduler: $LR_SCHEDULER_TYPE"
echo "Evaluation Strategy: $EVALUATION_STRATEGY"
echo "Save Strategy: $SAVE_STRATEGY"
echo "Save Steps: $SAVE_STEPS"
echo "Save every: $SAVE_EVERY_N_STEPS steps"
echo "Keep top: $SAVE_TOP_K checkpoints"
echo "Resume from checkpoint: $RESUME_FROM_CHECKPOINT"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Micro Batch Size: $MICRO_BATCH_SIZE"

echo "========================Entering NeMo Container============================"

# ============================================================================
# CONTAINER EXECUTION
# Launch distributed training inside NeMo container across all nodes
# ============================================================================
srun -l --container-image $CONTAINER_IMAGE \
        --container-mounts $CONTAINER_WORKSPACE_MOUNT:/workspace \
        --container-workdir /workspace \
        --no-container-mount-home \
        --export=ALL \
        --container-env=ALL \
        bash -c 'echo "---Inside NeMo Container: Node ID $SLURM_NODEID"; 
        echo "MASTER_ADDR: $MASTER_ADDR"; 
        echo "MASTER_PORT: $MASTER_PORT"; 
        echo "Setting up authentication...";
        wandb login "$WANDB_API_KEY"; 
        huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN"; 
        echo "Starting fresh NeMo 8B training with 1000-step checkpointing...";
        python tune_full_parameters_nemo.py \    # TODO: Update script path if different
            --model_name "$MODEL_NAME" \
            --processed_data_dir "$DATA_PATH" \
            --output_dir "$output_dir" \
            --max_epochs $MAX_EPOCHS \
            --num_devices $GPUS_PER_NODE \
            --accelerator gpu \
            --learning_rate $LEARNING_RATE \
            --weight_decay $WEIGHT_DECAY \
            --warmup_ratio $WARMUP_RATIO \
            --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
            --evaluation_strategy "$EVALUATION_STRATEGY" \
            --save_strategy "$SAVE_STRATEGY" \
            --save_steps $SAVE_STEPS \
            --global_batch_size $GLOBAL_BATCH_SIZE \
            --micro_batch_size $MICRO_BATCH_SIZE \
            --precision $PRECISION \
            --strategy $STRATEGY \
            --wandb_project "$WANDB_PROJECT" \
            --gradient_clip_val $GRADIENT_CLIP_VAL \
            --seed $SEED \
            --save_every_n_steps $SAVE_EVERY_N_STEPS \
            --save_top_k $SAVE_TOP_K \
            --resume_from_checkpoint $RESUME_FROM_CHECKPOINT'

# ============================================================================
# JOB COMPLETION SUMMARY
# Calculate and display total execution time
# ============================================================================
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "========================Job Completed============================"
echo "Job ended at: $(date)"
echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Total execution time (seconds): ${ELAPSED_TIME}"
echo "=============================================================="
