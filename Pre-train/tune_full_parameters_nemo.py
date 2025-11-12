#!/usr/bin/env python3
#Author: Jingmei Yang (jmyang@bu.edu)
"""
===============================================================================
NeMo AutoModel Full Parameter Fine-tuning Script
===============================================================================


===============================================================================
"""

import os
from typing import Optional
import torch
import warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from nemo import lightning as nl
from nemo.collections import llm
import random
import transformers
from dataclasses import dataclass, field
import numpy as np
import fiddle as fdl
import glob
from nemo.collections.llm.recipes import adam

@dataclass
class ModelArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-3.1-8B", metadata={"help": "Path or name of the base model."})
    strategy: Optional[str] = field(default="fsdp2", metadata={"help": "Distributed training strategy (ddp, fsdp2, etc)."})

@dataclass
class DataArguments:
    processed_data_dir: str = field(default="./Your Path/data", metadata={"help": "Directory of the processed dataset."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Maximum number of training samples to use (for faster training)."})
    max_val_samples: Optional[int] = field(default=None, metadata={"help": "Maximum number of validation samples to use."})

@dataclass
class TrainingArguments:
    output_dir: str = field(default="./output", metadata={"help": "Output directory for model checkpoints and logs."})
    max_steps: Optional[int] = field(default=None, metadata={"help": "Total number of training steps. If None, will use max_epochs."})
    max_epochs: Optional[int] = field(default=3, metadata={"help": "Total number of training epochs. Used when max_steps is None."})
    num_devices: int = field(default=1, metadata={"help": "Number of devices (GPUs) to use."})
    accelerator: str = field(default="gpu", metadata={"help": "Accelerator type (gpu, cpu, etc)."})
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate."})
    global_batch_size: int = field(default=96, metadata={"help": "Global batch size."})
    micro_batch_size: int = field(default=6, metadata={"help": "Micro batch size per device."})
    precision: str = field(default="bf16", metadata={"help": "Precision (bf16, fp16, etc)."})
    gradient_clip_val: float = field(default=1.0, metadata={"help": "Gradient clipping value."})
    seed: int = field(default=123, metadata={"help": "Random seed."})
    wandb_project: Optional[str] = field(default=None, metadata={"help": "Weights & Biases project name."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of gradient accumulation steps."})
    # CHECKPOINT ARGUMENTS
    save_every_n_steps: int = field(default=1000, metadata={"help": "Save checkpoint every N steps."})
    save_top_k: int = field(default=20, metadata={"help": "Save top K checkpoints."})
    resume_from_checkpoint: bool = field(default=False, metadata={"help": "Resume training from the latest checkpoint."})
    # HF-STYLE TRAINING ARGUMENTS
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay coefficient for regularization."})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Ratio of total training steps used for learning rate warmup."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate scheduler type (linear, cosine, constant, etc)."})
    evaluation_strategy: str = field(default="epoch", metadata={"help": "Evaluation strategy (steps, epoch, no)."})
    save_strategy: str = field(default="steps", metadata={"help": "Save strategy (steps, epoch, no)."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every N steps when save_strategy=steps."})



class CustomDataModule(pl.LightningDataModule):
    """Custom data module for LLM fine-tuning with NeMo AutoModel"""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer: AutoTokenizer,
        global_batch_size: int,
        micro_batch_size: int = 1,
        max_length: int = 2048,
        num_workers: int = 8,  # Increased workers for faster data loading
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,  # Number of batches loaded in advance by each worker
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # Only enable persistent_workers if num_workers > 0
        self.persistent_workers = persistent_workers and (num_workers > 0)
        # Only use prefetch_factor if num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Create data collator for language modeling (same as original HF script)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
        )
            
        self.datasets = load_from_disk(data_dir)
        print(f"Train dataset size: {len(self.datasets['train'])}")
        print(f"Test dataset size: {len(self.datasets['test'])}")

    def train_dataloader(self):
        """Create training dataloader"""
        dataloader_kwargs = {
            'batch_size': self.micro_batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'collate_fn': self.data_collator
        }
        # Only add prefetch_factor if num_workers > 0
        if self.prefetch_factor is not None:
            dataloader_kwargs['prefetch_factor'] = self.prefetch_factor
            
        return DataLoader(self.datasets['train'], **dataloader_kwargs)
    
    def val_dataloader(self):
        """Create validation dataloader"""
        return DataLoader(
            self.datasets['test'],
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.data_collator
        )


def make_strategy(strategy, model, devices, num_nodes, adapter_only=False):
    """
    Creates and returns a distributed training strategy based on the provided strategy type.

    Parameters:
        strategy (str): The name of the strategy ('ddp', 'fsdp2', or other).
        model: The model instance, which provides a method to create the HF checkpoint IO.
        devices (int): Number of devices per node.
        num_nodes (int): Number of compute nodes.
        adapter_only (bool, optional): Whether to save only adapter-related parameters in the checkpoint. Default is False.

    Returns:
        A PyTorch Lightning or custom distributed training strategy.
    """

    if strategy == 'ddp':  # Distributed Data Parallel (DDP) strategy
        return pl.strategies.DDPStrategy(
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
        )

    elif strategy == 'fsdp2':  # Fully Sharded Data Parallel (FSDP) v2 strategy
        return nl.FSDP2Strategy(
            data_parallel_size=devices * num_nodes,  # Defines total data parallel size
            tensor_parallel_size=1,  # No tensor parallelism
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
        )

    else:  # Default to single device strategy (useful for debugging or single-GPU training)
        return pl.strategies.SingleDeviceStrategy(
            device='cuda:0',  # Uses the first available CUDA device
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
        )


def calculate_batch_size_config(global_batch_size: int, micro_batch_size: int, num_devices: int) -> int:
    """
    Calculate gradient accumulation steps and validate batch size configuration.
    
    Args:
        global_batch_size: Target effective batch size
        micro_batch_size: Batch size per device
        num_devices: Number of devices/GPUs
    
    Returns:
        gradient_accumulation_steps: Required gradient accumulation steps
    """
    if global_batch_size % (micro_batch_size * num_devices) != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"(micro_batch_size × num_devices) = ({micro_batch_size} × {num_devices}) = {micro_batch_size * num_devices}"
        )
    
    gradient_accumulation_steps = global_batch_size // (micro_batch_size * num_devices)
    
    print(f"Batch Size Configuration:")
    print(f"  Global Batch Size: {global_batch_size}")
    print(f"  Micro Batch Size: {micro_batch_size}")
    print(f"  Number of Devices: {num_devices}")
    print(f"  Gradient Accumulation Steps: {gradient_accumulation_steps}")
    print(f"  Effective Batch Size per Device: {micro_batch_size * gradient_accumulation_steps}")
    
    return gradient_accumulation_steps

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest NeMo checkpoint in the output directory.
    NeMo saves checkpoints in: output_dir/nemo_automodel_finetuning/YYYY-MM-DD_HH-MM-SS/checkpoints/
    
    Args:
        output_dir: Directory to search for checkpoints
        
    Returns:
        Path to the latest checkpoint or None if no checkpoints found
    """
    # Look for NeMo's specific checkpoint structure
    nemo_pattern = os.path.join(output_dir, "nemo_automodel_finetuning", "*", "checkpoints", "*.ckpt")
    checkpoints = glob.glob(nemo_pattern, recursive=True)
    
    if not checkpoints:
        print(f"No NeMo checkpoints found in {output_dir}")
        print(f"Searched pattern: {nemo_pattern}")
        return None
    
    # Sort by step number in filename (more reliable than modification time)
    def extract_step(checkpoint_path):
        # Extract step number from filename like "nemo_automodel_finetuning--None=0.0000-epoch=0-step=739.ckpt"
        filename = os.path.basename(checkpoint_path)
        try:
            if "step=" in filename:
                step_part = filename.split("step=")[1].split(".")[0]
                return int(step_part)
            return 0
        except:
            return 0
    
    latest_checkpoint = max(checkpoints, key=extract_step)
    step_num = extract_step(latest_checkpoint)
    print(f"Found latest checkpoint at step {step_num}: {latest_checkpoint}")
    return latest_checkpoint

def main():
    # Step 1: Parse command-line arguments and initialize training configuration
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set random seeds
    seed = training_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("Start running the training script")
    print("Cuda check")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    
    # Print configuration summary
    print(f"\n=== Training Configuration ===")
    print(f"Model: {model_args.model_name}")
    print(f"Strategy: {model_args.strategy}")
    print(f"Global Batch Size: {training_args.global_batch_size}")
    print(f"Micro Batch Size: {training_args.micro_batch_size}")
    print(f"Learning Rate: {training_args.learning_rate}")
    print(f"Weight Decay: {training_args.weight_decay}")
    print(f"Warmup Ratio: {training_args.warmup_ratio}")
    print(f"LR Scheduler: {training_args.lr_scheduler_type}")
    print(f"Max Epochs: {training_args.max_epochs}")
    print(f"Max Steps: {training_args.max_steps}")
    print(f"Precision: {training_args.precision}")
    print(f"Output Dir: {training_args.output_dir}")
    print(f"Save Strategy: {training_args.save_strategy}")
    print(f"Save Steps: {training_args.save_steps}")
    print(f"Evaluation Strategy: {training_args.evaluation_strategy}")
    print(f"Save Every N Steps: {training_args.save_every_n_steps}")
    print(f"Save Top K: {training_args.save_top_k}")
    print(f"Resume from Checkpoint: {training_args.resume_from_checkpoint}")
    
    # Step 2: Calculate and validate batch size configuration for distributed training
    gradient_accumulation_steps = calculate_batch_size_config(
        training_args.global_batch_size,
        training_args.micro_batch_size,
        training_args.num_devices
    )
    # Update training_args with calculated value
    training_args.gradient_accumulation_steps = gradient_accumulation_steps

    # Step 3: Handle checkpoint resumption if requested
    resume_checkpoint_path = None
    if training_args.resume_from_checkpoint:
        resume_checkpoint_path = find_latest_checkpoint(training_args.output_dir)
        if resume_checkpoint_path:
            print(f"Will resume from checkpoint: {resume_checkpoint_path}")
        else:
            print("Resume requested but no checkpoint found. Starting from scratch.")

    # Step 4: Load and configure tokenizer with Hugging Face authentication
    hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')
    if hf_token:
        print("Using HuggingFace token from environment")
    else:
        print("Warning: No HuggingFace token found in environment")
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name,
            token=hf_token  # Add explicit token parameter
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Complete configuring tokenizer")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Make sure you have accepted the Llama model license and have a valid HF token")
        raise

    # Step 5: Create data module and combine OneHealth datasets (human, animal, plant, toxin)
    data_module = CustomDataModule(
        data_dir=data_args.processed_data_dir,
        tokenizer=tokenizer,
        global_batch_size=training_args.global_batch_size,
        micro_batch_size=training_args.micro_batch_size,
        max_length=4096
    )
    print("Data module created successfully")

    # Step 6: Initialize NeMo AutoModel for Causal Language Modeling
    print("Initializing NeMo HFAutoModelForCausalLM...")
    model = llm.HFAutoModelForCausalLM(model_name=model_args.model_name)
    
    # Note: NeMo AutoModel handles config and gradient checkpointing internally
    # No need to manually set use_cache or enable gradient checkpointing
    print("NeMo AutoModel initialized successfully")
    
    # Step 7: Configure distributed training strategy for multi-node setup
    num_nodes = int(os.environ.get('SLURM_NNODES', '1'))
    print(f"Number of nodes: {num_nodes}")
    print(f"Total devices: {training_args.num_devices * num_nodes}")

    # Create distributed strategy using make_strategy
    strategy = make_strategy(
        model_args.strategy,
        model,
        training_args.num_devices,
        num_nodes
    )

    # Step 8: Configure training duration and calculate optimization parameters
    if training_args.max_steps is not None:
        print(f"Training with max_steps: {training_args.max_steps}")
        max_steps = training_args.max_steps
        max_epochs = None
        total_steps = training_args.max_steps
    else:
        print(f"Training with max_epochs: {training_args.max_epochs}")
        max_steps = -1  # PyTorch Lightning uses -1 to indicate unlimited steps when using epochs
        max_epochs = training_args.max_epochs
        # Estimate total steps for warmup calculation
        train_dataset_size = len(data_module.datasets['train'])
        steps_per_epoch = train_dataset_size // training_args.global_batch_size
        total_steps = steps_per_epoch * training_args.max_epochs
        print(f"Estimated total steps: {total_steps} ({steps_per_epoch} steps/epoch)")
    
    # Calculate warmup steps
    warmup_steps = int(total_steps * training_args.warmup_ratio)
    print(f"Warmup steps: {warmup_steps} ({training_args.warmup_ratio:.1%} of total)")
    
    # Use save_steps if save_strategy is 'steps', otherwise use save_every_n_steps
    checkpoint_every_n_steps = training_args.save_steps if training_args.save_strategy == "steps" else training_args.save_every_n_steps
    print(f"Checkpointing every {checkpoint_every_n_steps} steps")
    
    # Configure validation frequency based on evaluation_strategy
    if training_args.evaluation_strategy == "steps":
        val_check_interval = checkpoint_every_n_steps  # Validate at same frequency as checkpointing
        limit_val_batches = 0.1  # Use more validation data when validating frequently
    elif training_args.evaluation_strategy == "epoch":
        val_check_interval = 1.0  # Validate once per epoch
        limit_val_batches = 0.05  # Use less validation data
    else:  # "no"
        val_check_interval = float('inf')  # Never validate
        limit_val_batches = 0
        print("Validation disabled")
    
    # Step 9: Configure checkpoint callback and trainer with robust settings
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(training_args.output_dir, "nemo_automodel_finetuning", "checkpoints"),
        filename="nemo_automodel_finetuning--None=0.0000-epoch={epoch}-step={step}",
        every_n_train_steps=checkpoint_every_n_steps,
        save_top_k=training_args.save_top_k,
        monitor="step",
        mode="max",
        save_last=True,  # Always save the last checkpoint
        verbose=True
    )
    
    # Configure trainer for NeMo API with HF-style settings
    trainer_config = nl.Trainer(
        devices=training_args.num_devices,
        max_steps=max_steps,
        max_epochs=max_epochs,
        accelerator=training_args.accelerator,
        strategy=strategy,
        precision=training_args.precision,
        gradient_clip_val=training_args.gradient_clip_val,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        log_every_n_steps=100,  # Log every 100 steps
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=0,
        use_distributed_sampler=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )
    print("Trainer configured successfully")
    
    # Logger setup
    nemo_logger = nl.NeMoLogger(
        log_dir=training_args.output_dir,
        name="nemo_automodel_finetuning",
        wandb=WandbLogger(project=training_args.wandb_project, name="nemo_automodel_finetuning") if training_args.wandb_project is not None else None,
    )

    # Step 10: Configure Adam optimizer with cosine annealing scheduler
    print(f"Configuring optimizer with lr={training_args.learning_rate}, weight_decay={training_args.weight_decay}")
    print(f"Configuring {training_args.lr_scheduler_type} scheduler with {warmup_steps} warmup steps")
    
    # Create optimizer configuration based on scheduler type
    optim_config = adam.pytorch_adam_with_cosine_annealing(
        max_lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        warmup_steps=warmup_steps,
    )
    optim_config = fdl.build(optim_config)
    
    # Step 11: Execute full parameter fine-tuning using NeMo API
    print("Starting fine-tuning with NeMo AutoModel API...")
    
    # Run fine-tuning using NeMo API
    llm.api.finetune(
        model=model,
        data=data_module,
        trainer=trainer_config,
        optim=optim_config,
        peft=None,  # Full fine-tuning (set to llm.peft.LoRA(...) for LoRA) 
        log=nemo_logger,  # Pass logger to NeMo API instead of trainer
        resume=resume_checkpoint_path,  # Resume from checkpoint if available
    )

    # Step 12: Save final model and tokenizer to output directory
    print("Saving model and tokenizer...")
    final_checkpoint = os.path.join(training_args.output_dir, "final_model")
    os.makedirs(final_checkpoint, exist_ok=True)
    tokenizer.save_pretrained(final_checkpoint)
    print(f"Model and tokenizer saved to: {final_checkpoint}")
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
    print("Complete all steps!!!")
