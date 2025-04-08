# LLM Fine-tuning Module

This module provides functionality for fine-tuning Large Language Models (LLMs) for polymer informatics tasks. It supports two main use cases:

1. LLM Classifier Training - For judging material property extractions
2. Property Prediction Training - For predicting polymer properties

## Environment Setup

Set up your environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
export MODEL_DIR="/path/to/output/directory"
export MODEL_CACHE_DIR="/path/to/cache/directory"
```

## Usage
### LLM Classifier Training

```bash
python -m llm_finetune.cli \
    --task_type classifier \
    --data_file path/to/your/data.jsonl \
    --num_epochs 25 \
    --alpha 16 \
    --rank 16 \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --use_4bit true \
    --batch_size 2 \
    --learning_rate 2e-4
```

### Property Prediction Training

```bash
python -m llm_finetune.cli \
    --task_type property_prediction \
    --property_abbr tg \
    --data_file path/to/your/data.csv \
    --num_epochs 30 \
    --alpha 16 \
    --rank 16 \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --use_4bit true \
    --batch_size 4 \
    --learning_rate 2e-4
```

## Configuration

### Parameters
- `--task_type`: Type of task ('classifier' or 'property_prediction')
- `--model_name`: Base model name or path
- `--use_4bit`: Whether to use 4-bit quantization
- `--num_epochs`: Number of training epochs
- `--alpha`: LoRA alpha parameter
- `--rank`: LoRA rank parameter
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate
- `--output_dir`: Base directory for outputs
- `--cache_dir`: Directory for caching models
- `--hf_token`: HuggingFace API token
-- `data_file`: Path to file containing training data

