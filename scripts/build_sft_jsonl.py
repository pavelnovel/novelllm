#!/usr/bin/env python3
"""
Converts clean corpus chunks into SFT (Supervised Fine-Tuning) JSONL format.
Each line in the output JSONL file will be a JSON object with 'text' field.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_chunks(chunks_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON chunks from the specified directory."""
    chunks = []
    for chunk_file in chunks_dir.glob('*.txt'):
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                if isinstance(chunk_data, list):
                    chunks.extend(chunk_data)
                else:
                    chunks.append(chunk_data)
        except Exception as e:
            logger.error(f"Error loading {chunk_file}: {e}")
    return chunks

def convert_to_sft_format(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert chunks to SFT format."""
    sft_data = []
    for chunk in chunks:
        try:
            # Extract text and timestamp
            text = chunk.get('text', '')
            timestamp = chunk.get('timestamp', '')
            
            # Skip empty or whitespace-only text
            if not text or not text.strip():
                continue
            
            # Create SFT format entry with instruction-tuning format
            sft_entry = {
                "instruction": "Write a social media post in Pavel's style.",
                "input": "",
                "output": text.strip()
            }
            sft_data.append(sft_entry)
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
    return sft_data

def save_jsonl(data: List[Dict[str, str]], output_file: Path):
    """Save data to JSONL file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(data)} entries to {output_file}")
        
        # Log sample output if we have any entries
        if data:
            logger.info(f"Sample SFT entry: {json.dumps(data[0], ensure_ascii=False)}")
    except Exception as e:
        logger.error(f"Error saving to {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Convert corpus chunks to SFT JSONL format')
    parser.add_argument('--input-dir', type=str, required=True,
                      help='Directory containing JSON chunks')
    parser.add_argument('--output-file', type=str, required=True,
                      help='Output JSONL file path')
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    # Ensure input directory exists
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process chunks
    logger.info(f"Loading chunks from {input_dir}")
    chunks = load_chunks(input_dir)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Convert to SFT format
    sft_data = convert_to_sft_format(chunks)
    logger.info(f"Converted {len(sft_data)} chunks to SFT format")
    
    # Save to JSONL
    save_jsonl(sft_data, output_file)

if __name__ == '__main__':
    main() 