#!/usr/bin/env python3
"""
Script to fix the tokenize_function in fine_tune.py to properly handle dataset structure.
"""

import sys
import re

def fix_tokenize_function(content: str) -> str:
    """Fix the tokenize_function in the content."""
    # Define the new tokenize_function
    new_tokenize_function = """def tokenize_function(examples):
    # Extract fields from examples
    instructions = examples["instruction"]
    inputs = examples.get("input", [""] * len(instructions))
    outputs = examples["output"]
    
    # Format each example
    texts = []
    for ins, inp, out in zip(instructions, inputs, outputs):
        prompt = ins + ("\\n" + inp if inp else "")
        src = prompt.strip()
        tgt = out.strip()
        text = f"{src}\\n{tgt}"
        texts.append(text)
    
    # Tokenize the formatted texts
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )"""

    # Find and replace the tokenize_function
    tokenize_pattern = re.compile(r"def tokenize_function\(examples\):.*?return tokenizer\(", re.DOTALL)
    match = tokenize_pattern.search(content)
    
    if match:
        # Keep the original tokenizer arguments
        tokenizer_args_start = match.end()
        closing_paren_depth = 1
        tokenizer_args_end = tokenizer_args_start
        
        for i in range(tokenizer_args_start, len(content)):
            if content[i] == '(':
                closing_paren_depth += 1
            elif content[i] == ')':
                closing_paren_depth -= 1
                if closing_paren_depth == 0:
                    tokenizer_args_end = i + 1
                    break
        
        tokenizer_args = content[tokenizer_args_start:tokenizer_args_end]
        
        # Replace the function but keep the original tokenizer arguments
        new_content = content[:match.start()] + new_tokenize_function + content[tokenizer_args_end:]
        return new_content
    else:
        # If we can't find the tokenize_function, return the original content
        return content

def main():
    # Read the input file
    content = sys.stdin.read()
    
    # Apply the fix
    fixed_content = fix_tokenize_function(content)
    
    # Print the result
    print(fixed_content)

if __name__ == "__main__":
    main() 