# PandemIQ Llama

[![Paper](https://img.shields.io/badge/AAAI-2026-blue)](https://huggingface.co/Paschalidis-NOC-Lab/PandemIQ-Llama)
[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/Paschalidis-NOC-Lab/PandemIQ-Llama)
[![BEACON](https://img.shields.io/badge/Platform-BEACON-red)](https://beaconbio.org)

PandemIQ Llama is a domain-adapted LLM for pandemic intelligence, built by continuous pre-training of Llama-3.1-8B on 5.8 billion tokens of pandemic-specific text. It powers the [BEACON platform](https://beaconbio.org) (Biothreats Emergence, Analysis and Communications Network), an open-source informal surveillance program designed to revolutionize global biothreats surveillance and response.

## Repository Structure
```
├── Data/          # Corpus construction and preprocessing
├── Pre-train/     # Continuous pre-training scripts
└── Fine-tune/     # Task-specific fine-tuning scripts
```

## Quick Start

### Download Model from Hugging Face

**Method 1: Using Transformers**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Paschalidis-NOC-Lab/PandemIQ-Llama",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Paschalidis-NOC-Lab/PandemIQ-Llama")
```

**Method 2: Using huggingface_hub**
```python
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="Paschalidis-NOC-Lab/PandemIQ-Llama",
    cache_dir="./models"
)
```

**Method 3: Using CLI**
```bash
huggingface-cli download Paschalidis-NOC-Lab/PandemIQ-Llama --local-dir ./PandemIQ-Llama
```

### Serve with vLLM
```python
from vllm import LLM, SamplingParams

# Initialize vLLM
llm = LLM(model="Paschalidis-NOC-Lab/PandemIQ-Llama")
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

# Generate
prompts = ["Input your question here"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```


### Fine-tune with LoRA
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Paschalidis-NOC-Lab/PandemIQ-Llama")
tokenizer = AutoTokenizer.from_pretrained("Paschalidis-NOC-Lab/PandemIQ-Llama")
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load dataset
train_dataset = load_dataset("json", data_files="train.json", split="train")

# Training configuration
training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    bf16=True
)

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    processing_class=tokenizer
)
trainer.train()
```

See `Fine-tune/` for complete examples including evaluation and multi-task setups.

## Pandemic Corpus

- **508,924 documents** from authoritative public health sources, scientific literature, and epidemiological databases
- **5.8 billion tokens** - the largest pandemic-specific corpus for LLM training
- **Dataset Access**: URLs available at [Pandemic-Corpus](https://huggingface.co/datasets/Paschalidis-NOC-Lab/Pandemic-Corpus)

*Note: We provide URLs rather than redistributing raw text to respect copyright and licensing.*

## Citation
```bibtex
@inproceedings{pandemiqllama2026,
  title={PandemIQ Llama: A Domain-Adapted Foundation Model for Enhanced Pandemic Intelligence},
  author={Yang, Jingmei and Talaei, Mahtab and Lassmann, Britta and Bhadelia, Nahid and Paschalidis, Ioannis Ch.},
  booktitle={AAAI},
  year={2026}
}
```

## Team

[Network Optimization and Control Lab, Boston University](https://sites.bu.edu/paschalidis/people/yannis-paschalidis/)

## License

MIT License. Model usage subject to Llama 3.1 Community License.