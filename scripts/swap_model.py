#!/usr/bin/env python3
"""
Safely replaces/updates the current production model with a new one.
Handles model backup and rollback in case of issues.

Expectations:
- model_dir should contain both model weights and config/tokenizer
- For rollbacks: --rollback 20240611_143855 will roll back to that exact timestamped backup
- For testing: Run this script after CI or fine-tune pipeline before production deployment
"""

import logging
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import json
import torch
import os
import fcntl
import tempfile
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelSwapper:
    def __init__(self, model_dir: Path, backup_dir: Path, use_symlink: bool = False):
        self.model_dir = model_dir
        self.backup_dir = backup_dir
        self.use_symlink = use_symlink
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.backup_dir / "swap_history.jsonl"
        self.lock_file = self.backup_dir / ".swap_lock"
        
    def acquire_lock(self) -> bool:
        """Acquire a lock file to prevent concurrent swaps."""
        try:
            with open(self.lock_file, 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
        except IOError:
            logger.error("Another swap operation is in progress")
            return False
            
    def release_lock(self):
        """Release the lock file."""
        try:
            os.remove(self.lock_file)
        except OSError:
            pass
            
    def atomic_copytree(self, src: Path, dst: Path):
        """Atomically copy a directory tree."""
        tmp_dst = dst.parent / (dst.name + "_tmp")
        try:
            if tmp_dst.exists():
                shutil.rmtree(tmp_dst)
            shutil.copytree(src, tmp_dst)
            if dst.exists():
                shutil.rmtree(dst)
            tmp_dst.rename(dst)
        except Exception as e:
            if tmp_dst.exists():
                shutil.rmtree(tmp_dst)
            raise e
            
    def log_history(self, action: str, details: Dict[str, Any]):
        """Log swap/rollback operations to history file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            **details
        }
        with open(self.history_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
            
    def create_backup(self) -> Path:
        """Create a backup of the current model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"model_backup_{timestamp}"
        
        logger.info(f"Creating backup at {backup_path}")
        self.atomic_copytree(self.model_dir, backup_path)
        
        # Save backup metadata
        metadata = {
            "timestamp": timestamp,
            "original_path": str(self.model_dir),
            "backup_path": str(backup_path)
        }
        with open(backup_path / "backup_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        self.log_history("backup", metadata)
        return backup_path
    
    def verify_model(self, model_path: Path, check_lora: bool = False) -> bool:
        """Verify that the model can be loaded and is valid."""
        try:
            logger.info(f"Verifying model at {model_path}")
            
            # Check for required files
            required_files = ["config.json", "pytorch_model.bin"]
            if check_lora:
                required_files.extend(["adapter_config.json", "adapter_model.bin"])
                
            for file in required_files:
                if not (model_path / file).exists():
                    logger.error(f"Required file {file} not found")
                    return False
            
            # Try loading the model
            if check_lora:
                config = PeftConfig.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name_or_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                model = PeftModel.from_pretrained(model, model_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            # Try loading the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Basic inference test
            test_input = "Hello, how are you?"
            inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=50)
            
            logger.info("Model verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False
    
    def swap_model(self, new_model_path: Path, check_lora: bool = False, dry_run: bool = False) -> bool:
        """Swap the current model with a new one."""
        if not self.acquire_lock():
            return False
            
        try:
            if dry_run:
                logger.info("Dry run - would perform the following operations:")
                logger.info(f"1. Create backup of {self.model_dir}")
                logger.info(f"2. Copy {new_model_path} to {self.model_dir}")
                return True
                
            # Create backup of current model
            backup_path = self.create_backup()
            
            # Verify new model
            if not self.verify_model(new_model_path, check_lora):
                logger.error("New model verification failed, aborting swap")
                return False
            
            # Remove current model and copy new one
            if self.use_symlink:
                # Create versioned directory
                version = datetime.now().strftime("v%Y%m%d_%H%M%S")
                version_dir = self.model_dir.parent / version
                self.atomic_copytree(new_model_path, version_dir)
                
                # Update symlink
                current_link = self.model_dir
                if current_link.exists():
                    current_link.unlink()
                current_link.symlink_to(version_dir)
            else:
                # Direct copy
                self.atomic_copytree(new_model_path, self.model_dir)
            
            # Verify the swap
            if not self.verify_model(self.model_dir, check_lora):
                logger.error("Model swap verification failed, rolling back")
                self.rollback(backup_path)
                return False
            
            self.log_history("swap", {
                "new_model": str(new_model_path),
                "backup": str(backup_path)
            })
            
            logger.info("Model swap completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during model swap: {e}")
            return False
        finally:
            self.release_lock()
    
    def rollback(self, backup_path: Path, dry_run: bool = False):
        """Rollback to a previous model version."""
        if not self.acquire_lock():
            return False
            
        try:
            if dry_run:
                logger.info("Dry run - would perform the following operations:")
                logger.info(f"1. Copy {backup_path} to {self.model_dir}")
                return True
                
            logger.info(f"Rolling back to backup at {backup_path}")
            
            # Restore from backup
            self.atomic_copytree(backup_path, self.model_dir)
            
            # Verify rollback
            if not self.verify_model(self.model_dir):
                logger.error("Rollback verification failed!")
                return False
            
            self.log_history("rollback", {
                "backup": str(backup_path)
            })
            
            logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False
        finally:
            self.release_lock()

def main():
    parser = argparse.ArgumentParser(description='Swap production model with a new one')
    parser.add_argument('--model-dir', type=str, required=True,
                      help='Directory of the current production model')
    parser.add_argument('--new-model', type=str,
                      help='Path to the new model')
    parser.add_argument('--backup-dir', type=str, required=True,
                      help='Directory to store model backups')
    parser.add_argument('--rollback', type=str,
                      help='Rollback to a specific backup (timestamp)')
    parser.add_argument('--use-symlink', action='store_true',
                      help='Use symlink for model versioning')
    parser.add_argument('--check-lora', action='store_true',
                      help='Verify LoRA adapter files')
    parser.add_argument('--dry-run', action='store_true',
                      help='Perform a dry run without making changes')
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    model_dir = Path(args.model_dir)
    backup_dir = Path(args.backup_dir)
    
    # Initialize swapper
    swapper = ModelSwapper(model_dir, backup_dir, args.use_symlink)
    
    if args.rollback:
        # Handle rollback
        backup_path = backup_dir / f"model_backup_{args.rollback}"
        if not backup_path.exists():
            logger.error(f"Backup {backup_path} does not exist")
            return
        
        success = swapper.rollback(backup_path, args.dry_run)
        if not success:
            logger.error("Rollback failed")
            return
        
    else:
        # Handle model swap
        if not args.new_model:
            logger.error("--new-model is required for swapping")
            return
            
        new_model = Path(args.new_model)
        if not new_model.exists():
            logger.error(f"New model {new_model} does not exist")
            return
        
        success = swapper.swap_model(new_model, args.check_lora, args.dry_run)
        if not success:
            logger.error("Model swap failed")
            return
    
    logger.info("Operation completed successfully")

if __name__ == '__main__':
    main() 