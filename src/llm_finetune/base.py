"""Base class for LLM finetuning operations."""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig

logger = logging.getLogger(__name__)

class BaseLLMFineTuner:
    """Base class for LLM finetuning operations."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        token: Optional[str] = None,
        use_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_quant_type: str = "nf4",
        use_nested_quant: bool = False,
        output_base_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the base LLM FineTuner.
        
        Args:
            model_name: Name or path of the base model
            token: HuggingFace API token
            use_4bit: Whether to use 4-bit quantization
            bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization
            bnb_4bit_quant_type: Quantization type for 4-bit
            use_nested_quant: Whether to use nested quantization
            output_base_dir: Base directory for outputs
            cache_dir: Directory for caching models
        """
        self.model_name = model_name
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("HuggingFace token must be provided or set in HF_TOKEN environment variable")
        
        self.use_4bit = use_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.use_nested_quant = use_nested_quant
        self.cache_dir = cache_dir or os.getenv("MODEL_CACHE_DIR", "/data/sonakshi/model_cache")
        
        # Set up output directories
        self.output_base_dir = Path(output_base_dir or os.getenv("MODEL_DIR", "models"))
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and tokenizer
        self._check_environment()
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _check_environment(self):
        """Check GPU environment."""
        assert torch.cuda.is_available(), "Failed to detect GPUs, make sure you set up cuda correctly!"
        logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")

        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info("=" * 80)
            logger.info("Your GPU supports bfloat16: accelerate training with bf16=True")
            logger.info("=" * 80)

    def _load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the model and tokenizer with specified configurations."""
        if self.use_4bit:
            logger.info("Using 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.use_4bit,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.use_nested_quant,
            )
        else:
            logger.info("Not using 4-bit quantization")
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=self.token,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=self.token,
            cache_dir=self.cache_dir
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer

    def _get_training_arguments(
        self,
        output_dir: str,
        num_epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
        save_steps: int,
        logging_steps: int
    ) -> TrainingArguments:
        """Get training arguments with common configuration."""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim="paged_adamw_32bit",
            save_steps=save_steps,
            logging_steps=logging_steps,
            save_total_limit=1,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=warmup_ratio,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="none"
        )

    def _get_peft_config(self, alpha: int, rank: int) -> LoraConfig:
        """Get PEFT configuration with common settings."""
        return LoraConfig(
            lora_alpha=alpha,
            lora_dropout=0.01,
            r=rank,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
            modules_to_save=["lm_head", "embed_token"]
        ) 