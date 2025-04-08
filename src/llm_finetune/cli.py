"""Command-line interface for LLM finetuning."""

import argparse
import logging
import os
from pathlib import Path

from .classifier import LLMClassifierFineTuner
from .property_prediction import PropertyPredictionFineTuner

logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Unified LLM Finetuning')
    
    # Common arguments
    parser.add_argument('--task_type', '-t', type=str, required=True,
                       choices=['classifier', 'property_prediction'],
                       help='Type of task to perform')
    parser.add_argument('--model_name', '-m', type=str,
                       default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help='Base model name or path')
    parser.add_argument('--use_4bit', '-u4bit', type=str, default='true',
                       help='Use 4-bit quantization')
    parser.add_argument('--num_epochs', '-e', type=int, required=True,
                       help='Number of training epochs')
    parser.add_argument('--alpha', '-a', type=int, required=True,
                       help='LoRA alpha parameter')
    parser.add_argument('--rank', '-r', type=int, required=True,
                       help='LoRA rank parameter')
    parser.add_argument('--batch_size', '-b', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--learning_rate', '-l', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--output_dir', '-o', type=str,
                       help='Base directory for outputs')
    parser.add_argument('--cache_dir', '-c', type=str,
                       help='Directory for caching models')
    parser.add_argument('--hf_token', type=str,
                       help='HuggingFace API token')
    parser.add_argument('--data_file', '-d', type=str,
                       help='Path to the training data file.')
    
    # Task-specific arguments
    parser.add_argument('--property_abbr', '-p', type=str,
                       choices=['tg', 'tm', 'td'],
                       help='Property abbreviation (for property prediction)')
    
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    setup_logging()
    args = parse_args()
    
    # Set environment variables from arguments
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    if args.output_dir:
        os.environ["MODEL_DIR"] = args.output_dir
    if args.cache_dir:
        os.environ["MODEL_CACHE_DIR"] = args.cache_dir
    
    try:
        # Initialize appropriate fine-tuner based on task type
        if args.task_type == 'classifier':
            if not args.data_file:
                raise ValueError("data_file is required for classifier task")
            finetuner = LLMClassifierFineTuner(
                model_name=args.model_name,
                use_4bit=args.use_4bit.lower() == 'true',
                output_base_dir=args.output_dir,
                cache_dir=args.cache_dir
            )
            model_dir = finetuner.train(
                data_file=args.data_file,
                num_epochs=args.num_epochs,
                alpha=args.alpha,
                rank=args.rank,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
        else:  # property_prediction
            if not args.property_abbr:
                raise ValueError("property_abbr is required for property prediction task")
            finetuner = PropertyPredictionFineTuner(
                model_name=args.model_name,
                use_4bit=args.use_4bit.lower() == 'true',
                output_base_dir=args.output_dir,
                cache_dir=args.cache_dir
            )
            model_dir = finetuner.train(
                property_abbr=args.property_abbr,
                data_file=args.data_file,
                num_epochs=args.num_epochs,
                alpha=args.alpha,
                rank=args.rank,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
        
        logger.info(f"Training completed successfully. Model saved to: {model_dir}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 