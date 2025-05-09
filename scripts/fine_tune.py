#!/usr/bin/env python3
"""
Runs LoRA/QLoRA fine-tuning on prepared data using PEFT library.
Supports both LoRA and QLoRA training modes.

Input data should be in SFT format with fields:
- instruction: The task instruction
- input: Optional input context
- output: The expected response
"""

import logging
import argparse
from pathlib import Path
from typing import Optional
import sys
import json

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import transformers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_device():
    """Check CUDA availability and warn if training on CPU."""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Training will be very slow on CPU!")
        return False
    
    logger.info(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    return True

def format_example(example):
    """Format SFT example into a single string."""
    # Check if example is a string (not a dictionary)
    if isinstance(example, str):
        logger.warning(f"Encountered example as string: {example}")
        # Handle it as needed - could return a default or skip
        return ""
    
    instruction = example.get("instruction", "")
    input_ = example.get("input", "")
    output = example.get("output", "")
    prompt = instruction + ("\n" + input_ if input_ else "")
    src = prompt.strip()
    tgt = output.strip()
    # For plain SFT: input = prompt, label = output
    return f"{src}\n{tgt}"

def load_model_and_tokenizer(
    model_name: str,
    use_qlora: bool = False,
    device_map: Optional[str] = "auto"
) -> tuple:
    """Load model and tokenizer with appropriate configuration."""
    logger.info(f"Loading model {model_name} with {'QLoRA' if use_qlora else 'LoRA'}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Determine optimal dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        logger.info("Using bfloat16 for better efficiency")
    else:
        dtype = torch.float16
        logger.info("Using float16 (bfloat16 not supported)")
    
    # Load model with appropriate configuration
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=dtype
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype
        )
    
    return model, tokenizer

def create_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None
) -> LoraConfig:
    """Create LoRA configuration."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

def prepare_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512
) -> torch.utils.data.Dataset:
    """Prepare dataset for SFT training."""
    logger.info(f"Loading dataset from {data_path}")
    
    # Load data directly from JSONL file
    raw_data = []
    with open(data_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                raw_data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing JSON line: {e}")
                continue
    
    # Create dataset from list of dictionaries
    from datasets import Dataset
    dataset = Dataset.from_list(raw_data)
    
    def format_example(example):
        """Format SFT example into a single string."""
        # Ensure example is a dictionary
        if not isinstance(example, dict):
            logger.warning(f"Invalid example format: {example}")
            return ""
        
        instruction = example.get("instruction", "")
        input_ = example.get("input", "")
        output = example.get("output", "")
        
        if not instruction or not output:
            logger.warning(f"Missing required fields in example: {example}")
            return ""
            
        prompt = instruction + ("\n" + input_ if input_ else "")
        src = prompt.strip()
        tgt = output.strip()
        return f"{src}\n{tgt}"

    def tokenize_function(examples):
        # Extract fields from examples
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]
        
        # Format each example
        texts = []
        for ins, inp, out in zip(instructions, inputs, outputs):
            prompt = ins + ("\n" + inp if inp else "")
            src = prompt.strip()
            tgt = out.strip()
            text = f"{src}\n{tgt}"
            texts.append(text)
        
        # Tokenize the formatted texts
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )

    # Map doesn't work with list-of-dict w/ wrong columns; force columns
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset

def main():
    try:
        parser = argparse.ArgumentParser(description='Fine-tune model using LoRA/QLoRA')
        parser.add_argument('--model-name', type=str, required=True,
                          help='Base model name or path')
        parser.add_argument('--data-path', type=str, required=True,
                          help='Path to JSONL training data')
        parser.add_argument('--output-dir', type=str, required=True,
                          help='Directory to save the fine-tuned model')
        parser.add_argument('--use-qlora', action='store_true',
                          help='Use QLoRA instead of LoRA')
        parser.add_argument('--lora-r', type=int, default=8,
                          help='LoRA rank')
        parser.add_argument('--lora-alpha', type=int, default=16,
                          help='LoRA alpha')
        parser.add_argument('--lora-dropout', type=float, default=0.05,
                          help='LoRA dropout')
        parser.add_argument('--max-length', type=int, default=512,
                          help='Maximum sequence length')
        parser.add_argument('--batch-size', type=int, default=4,
                          help='Training batch size')
        parser.add_argument('--eval-batch-size', type=int, default=4,
                          help='Evaluation batch size')
        parser.add_argument('--num-epochs', type=int, default=3,
                          help='Number of training epochs')
        parser.add_argument('--learning-rate', type=float, default=2e-4,
                          help='Learning rate')
        parser.add_argument('--seed', type=int, default=42,
                          help='Random seed for reproducibility')
        parser.add_argument('--eval-data-path', type=str,
                          help='Path to evaluation data (optional)')
        parser.add_argument('--push-to-hub', action='store_true',
                          help='Push model to HuggingFace Hub')
        parser.add_argument('--hub-model-id', type=str,
                          help='Model ID for HuggingFace Hub')
        
        args = parser.parse_args()
        
        # Set random seed
        set_seed(args.seed)
        
        # Check device
        has_cuda = check_device()
        if not has_cuda:
            logger.warning("Consider using a machine with GPU for faster training")
        
        # Log versions for reproducibility
        logger.info(f"Transformers version: {transformers.__version__}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            args.model_name,
            use_qlora=args.use_qlora
        )
        
        # Create and apply LoRA configuration
        lora_config = create_lora_config(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        model = get_peft_model(model, lora_config)
        
        # Prepare datasets
        train_dataset = prepare_dataset(
            args.data_path,
            tokenizer,
            max_length=args.max_length
        )
        
        eval_dataset = None
        if args.eval_data_path:
            eval_dataset = prepare_dataset(
                args.eval_data_path,
                tokenizer,
                max_length=args.max_length
            )
        
        # Calculate optimal logging steps
        logging_steps = max(1, len(train_dataset) // 1000)
        logger.info(f"Setting logging steps to {logging_steps} based on dataset size")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=logging_steps,
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id if args.push_to_hub else None,
            dataloader_pin_memory=True,
            dataloader_num_workers=4
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Save model (only LoRA adapters)
        logger.info(f"Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 