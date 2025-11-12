#!/usr/bin/env python3
#Author: Jingmei Yang (jmyang@bu.edu)

import os
import json
import argparse
from datasets import load_dataset
import random
import warnings
from utility import load_text_file, save2jsonl, load_json_file


def format_example(example, system_message):
    conversation = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": f"Context: {example['context']}\n\nQuestion: {example['question']}\nInstruction: {example['instruction']}\nQuestion: {example['question']}\n\nResponse: "
        },
        {
            "role": "assistant",
            "content": json.dumps({"Answer": example["answers"]})
        }
    ]
    return {"messages": conversation}



def main():
    parser = argparse.ArgumentParser(description="Script for creating fine-tuning data.")
    parser.add_argument("--test_data", type=str, default="Data/your path/processed_test.json",
                        help="Path to the test dataset file (JSON).")
    parser.add_argument("--system_file", type=str, default="your path/system_message.txt",
                        help="Path to the system message text file.")
    parser.add_argument("--output_file", type=str, default="your path/sft_chat_test.json",
                        help="Path to the output JSONL file.")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    system_message = load_text_file(args.system_file)
    test_dataset = load_dataset("json", data_files=args.test_data, split="train")

    def transform_fn(example):
        return format_example(example, system_message)

    transformed_dataset = test_dataset.map(transform_fn, num_proc=40,remove_columns=test_dataset.column_names)
    transformed_dataset.to_json(args.output_file, orient="records", lines=True, force_ascii=False)
    print(f"Prompt-completion dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()