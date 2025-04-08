"""LLM Classifier training module."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from datasets import load_dataset
from peft import get_peft_model
from trl import SFTTrainer

from .base import BaseLLMFineTuner

logger = logging.getLogger(__name__)

class LLMClassifierFineTuner(BaseLLMFineTuner):
    """Specialized class for LLM classifier training."""
    
    def format_instruction(self, row: Dict[str, Any], dataset_field: str) -> Dict[str, Any]:
        """Format instruction for LLM classifier training."""
        row[dataset_field] = f"<s>USER: {self._format_data(row)}\nASST: {self._format_asst_response(row)}</s>"
        return row

    def _format_data(self, data: Dict[str, Any]) -> str:
        """Format input data for LLM classifier."""
        return f"""
Given the extracted material name and property value, determine whether they have been correctly derived from the provided context and assign a score between -1 and 1. Also give the corrected extracted material name and property value.

### Extracted Value:
{data["input"]["extracted"]}

### Context:
{data["input"]["context"]}

### Thinking Process:
1. Material Name Check – Does the extracted material name appear in the context exactly, partially, or not at all?
2. Tg Value Check – Is the extracted Tg value explicitly stated in the context?
3. Scoring Criteria:
   - 1: Fully correct (material name and Tg value match the context exactly).
   - 0.75: Partially correct (minor discrepancies in material name or Tg value).
   - 0.5: Incorrect material name or Tg value.
   - -0.5: One of them is hallucinated (not found in the context).
   - -1: Both are hallucinated (completely incorrect extraction).
   
### Output:
Think through the reasoning steps and then **respond only in JSON format as follows**:
{{"score": "<numeric_score>", "justification": "<brief explanation for the score>", "correct_response": {{"material_name": "<correct_material_name>", "property_value": "<correct_property_value>"}}}}
"""

    def _format_asst_response(self, data: Dict[str, Any]) -> str:
        """Format assistant response for LLM classifier."""
        return f'{{"score":{data["response"]["score"]}, "justification":"{data["response"]["justification"]}", "correct_response":{data["input"]["ground_truth"]}}}'

    def train(
        self,
        data_file: str,
        num_epochs: int,
        alpha: int,
        rank: int,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 2,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.001,
        warmup_ratio: float = 0.03,
        save_steps: int = 100,
        logging_steps: int = 100
    ) -> str:
        """Train the LLM classifier model.
        
        Args:
            data_file: Path to the training data file (JSONL)
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
        # Load and prepare dataset
        jsonl_dataset = load_dataset('json', data_files=data_file)['train']
        logger.info(f"Loaded {len(jsonl_dataset)} examples from {data_file}")
        
        dataset = jsonl_dataset.map(lambda row: self.format_instruction(row, "prompt"))
        
        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.output_base_dir / f"llm_judge_{timestamp}"
        output_dir = self.output_base_dir / f"llm_judge_checkpoints_{timestamp}"
        
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
        logger.info("Starting LLM classifier training")
        trainer.train()
        
        # Save trained model
        trainer.model.save_pretrained(str(model_dir))
        logger.info(f"Model saved to {model_dir}")
        
        return str(model_dir) 