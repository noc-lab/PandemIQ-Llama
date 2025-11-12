#!/usr/bin/env python3
#Author: Jingmei Yang (jmyang@bu.edu)

import argparse
import os
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from utility import load_json_file
import warnings
import wandb
from accelerate import Accelerator
from transformers import EarlyStoppingCallback


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def parse_arguments():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning via config file.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to JSON or YAML config file.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    config = load_json_file(args.config_path)
    accelerator = Accelerator()
    if accelerator.is_main_process:
        for key, value in config.items():
            print(f"{key}: {value}")
        wandb.init(project="your project name", name=config["run_name"], config=config)


    OUTDIR = os.path.join(config["output_dir"],
                          config["run_name"],
                          config["model_path"],
                          str(config["r"]))


    train_dataset = load_dataset("json", data_files=config["your path/train_data_path"], split="train")
    train_dataset = train_dataset.shuffle(seed)
    eval_dataset = load_dataset("json", data_files=config["your path/eval_data_path"], split="train")

    base_model = AutoModelForCausalLM.from_pretrained(config["model_path"])

    base_model.config.use_cache = False
    base_model.config.gradient_checkpointing = True

  
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], use_fast=False)
    pad_token_str = "<|finetune_right_pad_id|>"

    if pad_token_str not in tokenizer.get_vocab():
        tokenizer.add_tokens([pad_token_str])
        base_model.resize_token_embeddings(len(tokenizer))

    tokenizer.pad_token = pad_token_str
    base_model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 4096


    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        inference_mode=False,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        output_dir=OUTDIR,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        eval_strategy=config["evaluation_strategy"],
        save_strategy=config["save_strategy"],
        save_total_limit=config["save_total_limit"],
        logging_strategy=config["logging_strategy"],
        run_name=config["run_name"],
        bf16=True,
        max_grad_norm=config.get("max_grad_norm", 1),
        report_to="wandb",
        push_to_hub=False,
        dataloader_num_workers=10,
        dataloader_prefetch_factor=4,
        save_on_each_node=False,
        disable_tqdm=False,          
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        processing_class=tokenizer,
    )


    # Train
    trainer.train()


if __name__ == "__main__":
    main()