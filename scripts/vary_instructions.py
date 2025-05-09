import json
import random
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define instruction templates
INSTRUCTION_TEMPLATES = [
    "Write a LinkedIn post in Pavel's style",
    "Create a professional social media post in Pavel's style",
    "Compose a thought-provoking LinkedIn post in Pavel's style",
    "Draft a business-focused social media post in Pavel's style",
    "Write an engaging LinkedIn post in Pavel's style",
    "Create a professional insight in Pavel's style",
    "Compose a strategic business post in Pavel's style",
    "Write a marketing-focused LinkedIn post in Pavel's style",
    "Draft a thought leadership piece in Pavel's style",
    "Create a business analysis post in Pavel's style",
    "Write a business reflection in Pavel's style",
    "Compose a professional observation in Pavel's style",
    "Create a strategic insight in Pavel's style",
    "Write a business perspective in Pavel's style",
    "Draft a professional thought in Pavel's style"
]

def generate_varied_instructions(input_file: str, output_file: str) -> None:
    """
    Generate varied instructions for the training data while maintaining Pavel's style.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output JSONL file
    """
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        # Generate varied instructions
        varied_data = []
        for entry in data:
            # Select random template
            new_instruction = random.choice(INSTRUCTION_TEMPLATES)
            
            # Create new entry with varied instruction
            new_entry = {
                "instruction": new_instruction,
                "input": entry.get("input", ""),
                "output": entry.get("output", "")
            }
            varied_data.append(new_entry)
        
        # Write the varied data to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in varied_data:
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Successfully generated varied instructions. Output written to {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating varied instructions: {e}")

if __name__ == "__main__":
    input_file = "sft_data.jsonl"
    output_file = "sft_data_varied.jsonl"
    generate_varied_instructions(input_file, output_file) 