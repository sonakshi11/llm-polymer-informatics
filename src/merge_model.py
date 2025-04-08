#!/usr/bin/env python3
"""
Model Merging Script for Property Prediction
This script merges a base model with its LoRA adapters to create a standalone model.
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name, token, use_4bit=False):
    """Load the base model and tokenizer."""
    if use_4bit:
        logger.info("Using 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
    else:
        logger.info("Not using 4-bit quantization")
        bnb_config = None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=token,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=token
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Merge base model with LoRA adapters')
    parser.add_argument('--base_model', '-b', type=str, required=True,
                       help='Path to the base model')
    parser.add_argument('--adapter_path', '-a', type=str, required=True,
                       help='Path to the LoRA adapter')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                       help='Directory to save the merged model')
    parser.add_argument('--token', '-t', type=str, required=True,
                       help='Hugging Face token')
    parser.add_argument('--use_4bit', '-u', type=str, default="false",
                       help='Use 4-bit quantization')
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base model and tokenizer
        logger.info(f"Loading base model from {args.base_model}")
        model, tokenizer = load_model_and_tokenizer(
            args.base_model,
            args.token,
            use_4bit=args.use_4bit.lower() == "true"
        )
        
        # Load and apply LoRA adapter
        logger.info(f"Loading LoRA adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        
        # Merge model
        logger.info("Merging model with LoRA adapter")
        model = model.merge_and_unload()
        
        # Save merged model
        # Get model name from adapter path
        model_name = os.path.basename(args.adapter_path)
        model_output_dir = output_dir / f"merged-{model_name}"
        
        logger.info(f"Saving merged model to {model_output_dir}")
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_output_dir))
        tokenizer.save_pretrained(str(model_output_dir))
        
        logger.info("Model merging completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model merging: {str(e)}")
        raise

if __name__ == "__main__":
    main() 