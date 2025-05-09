#!/usr/bin/env python3
"""
Debug script to analyze dataset structure and formatting.
"""

import json
from pathlib import Path
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_dataset(data_path: str):
    """Analyze the dataset structure and content."""
    logger.info(f"Analyzing dataset at: {data_path}")
    
    # Read and print the first few examples
    logger.info("\nFirst 3 examples from the raw file:")
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 3:
                try:
                    item = json.loads(line.strip())
                    logger.info(f"Example {i+1}: {item}")
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON in line {i+1}: {line.strip()}")
            else:
                break

    # Print expected format
    logger.info("\nExpected formatting output:")
    example = {
        "instruction": "Write a social media post in Pavel's style.",
        "input": "",
        "output": "Our nostalgia is so strong that we use our iPhones to recreate an old-school Nokia phone layout to play Snake"
    }
    src = example["instruction"] + ("\n" + example["input"] if example["input"] else "")
    tgt = example["output"]
    logger.info(f"Source: {src}")
    logger.info(f"Target: {tgt}")
    logger.info(f"Combined: {src}\n{tgt}")

    # Test dataset creation with datasets library
    try:
        from datasets import Dataset
        
        logger.info("\nTesting Dataset creation:")
        # Read all data
        raw_data = []
        with open(data_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    raw_data.append(item)
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON: {line.strip()}")
        
        logger.info(f"Read {len(raw_data)} examples")
        dataset = Dataset.from_list(raw_data)
        logger.info(f"Dataset created with {len(dataset)} examples")
        
        # Test format_example function
        def format_example(example):
            if not isinstance(example, dict):
                logger.warning(f"Warning: Example is not a dict: {type(example)}, {example}")
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
        
        # Test with first example
        if len(raw_data) > 0:
            formatted = format_example(raw_data[0])
            logger.info(f"\nFormatted first example: {formatted}")
        
        # Test dataset mapping
        def tokenize_function_test(examples):
            logger.info(f"examples keys: {examples.keys()}")
            logger.info(f"First few values of the first key: {list(examples.values())[0][:2]}")
            return examples
        
        # Try mapping
        mapped_dataset = dataset.map(tokenize_function_test, batched=True)
        logger.info(f"Mapped dataset has {len(mapped_dataset)} examples")
        
    except Exception as e:
        logger.error(f"Error testing dataset: {e}")

def main():
    if len(sys.argv) != 2:
        logger.error("Usage: python debug_dataset.py <path_to_dataset>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    analyze_dataset(data_path)

if __name__ == "__main__":
    main() 