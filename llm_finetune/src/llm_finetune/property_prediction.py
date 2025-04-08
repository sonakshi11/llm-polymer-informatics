"""Property prediction training module."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from datasets import Dataset, load_dataset
from peft import get_peft_model
from trl import SFTTrainer

from .base import BaseLLMFineTuner

logger = logging.getLogger(__name__)

class PropertyPredictionFineTuner(BaseLLMFineTuner):
    """Specialized class for property prediction training."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.property_mapping = {
            'tg': 'Glass Transition Temperature',
            'tm': 'Melting Temperature',
            'td': 'Decomposition Temperature'
        }
    def format_instruction(self, row: Dict[str, Any], dataset_field: str, property_name: str) -> Dict[str, Any]:
        """Format instruction for property prediction training."""
        row[dataset_field] = f"""<s>USER: If the SMILES of a polymer is {row['canonical_smiles']}, what is its {property_name}?
ASST: smiles: {row['canonical_smiles']}, {property_name}: {row['value']} {row['unit']}</s>"""
        return row

    def _load_dataset(self, data_file: str, property_name: str) -> Dataset:
        """Load and prepare the dataset for training.
        
        Args:
            data_file: Path to the data file (CSV or JSONL)
            property_name: Name of the property being predicted
            
        Returns:
            Dataset: Prepared dataset for training
        """
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
            required_columns = ["canonical_smiles", "value", "unit"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}")
            
            dataset = Dataset.from_pandas(df)
        else:
            dataset = load_dataset('json', data_files=data_file)['train']
        
        return dataset.map(lambda row: self.format_instruction(row, "prompt", property_name))

    def train(
        self,
        property_abbr: str,
        data_file: str,
        num_epochs: int,
        alpha: int,
        rank: int,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.001,
        warmup_ratio: float = 0.03,
        save_steps: int = 100,
        logging_steps: int = 100
    ) -> str:
        """Train the property prediction model.
        
        Args:
            property_abbr: Property abbreviation (tg, tm, td)
            data_file: Path to the training data file
            num_epochs: Number of training epochs
            alpha: LoRA alpha parameter
            rank: LoRA rank parameter
            batch_size: Training batch size
            gradient_accumulation_steps: Number of gradient accumulation steps
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_ratio: Warmup ratio
            save_steps: Number of steps between checkpoints
            logging_steps: Number of steps between logging
            
        Returns:
            str: Path to the saved model
        """
        if property_abbr not in self.property_mapping:
            raise ValueError(f"Invalid property abbreviation. Must be one of: {list(self.property_mapping.keys())}")
        
        property_name = self.property_mapping[property_abbr]
        
        # Load and prepare dataset
        dataset = self._load_dataset(data_file, property_name)
        logger.info(f"Loaded {len(dataset)} examples from {data_file}")
        
        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.output_base_dir / f"property_prediction_{property_abbr}_{timestamp}"
        output_dir = self.output_base_dir / f"property_prediction_checkpoints_{property_abbr}_{timestamp}"
        
        # Get configurations
        peft_config = self._get_peft_config(alpha, rank)
        training_arguments = self._get_training_arguments(
            str(output_dir), num_epochs, batch_size, gradient_accumulation_steps,
            learning_rate, weight_decay, warmup_ratio, save_steps, logging_steps
        )
        
        # Add LoRA modules
        model = get_peft_model(self.model, peft_config)
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="prompt",
            max_seq_length=8192,
            tokenizer=self.tokenizer,
            args=training_arguments,
            packing=False,
        )
        
        # Train
        logger.info(f"Starting property prediction training for {property_name}")
        trainer.train()
        
        # Save trained model
        trainer.model.save_pretrained(str(model_dir))
        logger.info(f"Model saved to {model_dir}")
        
        return str(model_dir) 