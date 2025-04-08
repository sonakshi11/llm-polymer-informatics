# LLM Fine-tuning for Polymer Informatics

This repository contains scripts for fine-tuning and using Large Language Models (LLMs) for polymer informatics tasks. The project includes tools for model training, model merging and inference.


## Installation

1. Clone the repository:
```bash
git clone https://github.com/sonakshi11/llm-polymer-informatics.git
cd property-prediction
```

2. Install dependencies:
```bash
source env.sh
pip install -r requirements.txt
```

## Scripts

### 1. Fine-tuning 

The fine-tuning module supports two main use cases:

1. LLM Classifier Training - For judging material property extractions
2. Property Prediction Training - For predicting polymer properties


### 2. Model Merging (`merge_model.py`)

This script merges a base model with its LoRA adapters to create a standalone model.

```bash
python merge_model.py \
    --base_model <base_model_path> \
    --adapter_path <adapter_path> \
    --output_dir <output_directory> \
    --token <huggingface_token> \
    --use_4bit <true/false>
```

### 3. Inference (`inference.py`)

This script runs inference with fine-tuned models for property prediction/ data validation tasks.

```bash
python inference.py \
    --model_name <model_name> \
    --token <huggingface_token> \
    --finetuned_model_path <finetuned_model_path> \
    --test_jsonl_file <test_file> \
    --output_dir <output_directory> \
    --temperature <temperature> \
    --condition <condition> \
    --use_4bit <true/false>
```


## Data Format

### LLM Classifier Data (JSONL)
Each line should be a JSON object with the following structure:
```json
{
    "input": {
        "extracted": "extracted material and property",
        "context": "original context",
        "ground_truth": {"material_name": "...", "property_value": "..."}
    },
    "response": {
        "score": 1.0,
        "justification": "explanation"
    }
}
```

### Property Prediction Data (CSV)
The CSV file should contain the following columns:
- `canonical_smiles`: SMILES representation of the polymer
- `value`: Property value
- `unit`: Unit of measurement

