#!/usr/bin/env python3
"""
Inference Script for Property Prediction
This script runs inference with fine-tuned LLaMA models for property prediction tasks.
"""

import os
import sys
import torch
import logging
import argparse
import pandas as pd
from pathlib import Path
from time import perf_counter
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
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

def load_model_and_tokenizer(model_name, token, use_4bit=True):
    """Load the model and tokenizer."""
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

def generate_response(model, tokenizer, prompt, temperature=0.5):
    """Generate response for a given prompt."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        
        generation_config = GenerationConfig(
            penalty_alpha=0.6,
            do_sample=True,
            top_k=3,
            temperature=temperature,
            repetition_penalty=1.2,
            max_new_tokens=12,
            pad_token_id=tokenizer.eos_token_id
        )

        start_time = perf_counter()
        outputs = model.generate(**inputs, generation_config=generation_config)
        generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_time = perf_counter() - start_time

        return generated_response, round(output_time, 2)
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

def format_prompt(example, dataset_field="prompt"):
    """Format example for inference."""
    try:
        system_message = example['messages'][0]['content']
        user_message = example['messages'][1]['content']
        assistant_message = example['messages'][2]['content']
        formatted_prompt = f"<s>USER: {user_message}\nASST: "
        
        example[dataset_field] = formatted_prompt
        example['target'] = assistant_message
        return example
    except Exception as e:
        logger.error(f"Error formatting prompt: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned model')
    parser.add_argument('--model_name', '-m', type=str, 
                       default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help='Name or path of the base model')
    parser.add_argument('--token', '-t', type=str, required=True,
                       help='Hugging Face token for model access')
    parser.add_argument('--finetuned_model_path', '-fp', type=str, required=False,
                       help='Pah to the merged finetuned model')
    parser.add_argument('--test_jsonl_file', '-j', type=str, required=False,
                       help='Path to input JSONL file')
    parser.add_argument('--output_dir', '-o', type=str, 
                       default="finetuned-models-responses/",
                       help='Directory to save response outputs')
    parser.add_argument('--csv_file', '-c', type=str, default=None,
                       help='Path to input CSV file (optional)')
    parser.add_argument('--temperature', '-temp', type=float, default=0.5,
                       help='Temperature for inference')
    parser.add_argument('--condition', '-cond', type=str, default=None,
                       help='Condition for inference')
    parser.add_argument('--use_4bit', '-u', type=str, default="true",
                       help='Use 4-bit quantization')
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine adapter path
        adapter_path = None
        if hasattr(args, 'finetuned_model_path') and args.finetuned_model_path is not None:
            adapter_path = args.finetuned_model_path

        # If finetuned model path is provided, separate model dir and name
        if adapter_path is not None:
            adapter_dir = os.path.dirname(adapter_path)
            adapter_name = os.path.basename(adapter_path)
            logger.info(f"Using adapter from directory: {adapter_dir}")
            logger.info(f"Adapter name: {adapter_name}")
            
            # Update adapter path to use separated components
            adapter_path = os.path.join(adapter_dir, adapter_name)
        
        # Create output filename
        try:
            test_resp_file = os.path.join(
                output_dir, 
                f"{'base' if adapter_path is None else adapter_name}_T{str(args.temperature).replace('.', '')}_test_resp{'_' + args.condition if args.condition is not None else ''}.csv"
            )
        except:
            test_resp_file = os.path.join(
                output_dir, 
                f"{'base' if adapter_path is None else adapter_name}_T{str(args.temperature).replace('.', '')}_test_resp{'_' + args.condition if args.condition is not None else ''}.csv"
            )
            
        # Load model and tokenizer
        logger.info(f"Loading model from {adapter_path if adapter_path else args.model_name}")
        model, tokenizer = load_model_and_tokenizer(
            adapter_path if adapter_path else args.model_name,
            args.token,
            use_4bit=args.use_4bit.lower() == "true"
        )
        
        # Load dataset
        if args.csv_file is not None:
            logger.info(f"Loading CSV file: {args.csv_file}")
            test_df = pd.read_csv(args.csv_file)
        else:
            logger.info(f"Loading JSONL file: {args.test_jsonl_file}")
            jsonl_dataset = load_dataset('json', data_files=args.test_jsonl_file)['train']
            dataset = jsonl_dataset.map(format_prompt)
            test_df = pd.DataFrame(dataset)[['prompt', 'target']]
            
        logger.info(f"Loaded {len(test_df)} examples for inference")
        
        # Run inference
        logger.info("Starting inference...")
        for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            generated_response, response_time = generate_response(
                model, 
                tokenizer, 
                row['prompt'], 
                args.temperature
            )
            test_df.at[index, 'generated_response'] = generated_response
            test_df.at[index, 'response_time'] = response_time
            
        # Save results
        logger.info(f"Saving results to {test_resp_file}")
        test_df.to_csv(test_resp_file, index=False)
        logger.info("Inference completed successfully")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main() 