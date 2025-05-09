from pathlib import Path
import re
import json
import logging
from typing import List, Union, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ChunkingConfig:
    """Configuration for the chunking process."""
    corpus_dir: str = "corpus"
    chunks_root: str = "chunks"
    separator: str = "---"
    chunk_prefix: str = "chunk_"
    chunk_suffix: str = ".txt"
    encoding: str = "utf-8"

class DocumentChunker:
    """Handles document chunking and processing."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize the document chunker.
        
        Args:
            config: Configuration for the chunking process
        """
        self.config = config
        self.corpus_dir = Path(config.corpus_dir)
        self.chunks_root = Path(config.chunks_root)
    
    def _create_output_dir(self, file_path: Path) -> Path:
        """Create output directory for chunks.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            Path to the output directory
        """
        out_dir = self.chunks_root / file_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    
    def _process_json_file(self, file_path: Path) -> List[str]:
        """Process a JSON file into chunks.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of chunked content
        """
        try:
            raw_text = file_path.read_text(encoding=self.config.encoding)
            json_data = json.loads(raw_text)
            return [json.dumps(item, ensure_ascii=False) for item in json_data]
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {file_path.name}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error processing JSON file {file_path.name}: {str(e)}")
            return []
    
    def _process_markdown_file(self, file_path: Path) -> List[str]:
        """Process a Markdown file into chunks.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            List of chunked content
        """
        try:
            raw_text = file_path.read_text(encoding=self.config.encoding)
            # Use regex to split on lines that contain only the separator
            return re.split(
                f"^{self.config.separator}$",
                raw_text,
                flags=re.MULTILINE
            )
        except Exception as e:
            logger.error(f"Error processing Markdown file {file_path.name}: {str(e)}")
            return []
    
    def _save_chunks(self, chunks: List[str], out_dir: Path) -> int:
        """Save chunks to files.
        
        Args:
            chunks: List of chunked content
            out_dir: Directory to save chunks in
            
        Returns:
            Number of chunks saved
        """
        chunk_count = 0
        for i, chunk in enumerate(chunks):
            cleaned = chunk.strip()
            if cleaned:
                chunk_file = out_dir / f"{self.config.chunk_prefix}{chunk_count:03}{self.config.chunk_suffix}"
                chunk_file.write_text(cleaned, encoding=self.config.encoding)
                chunk_count += 1
        return chunk_count
    
    def process_file(self, file_path: Path) -> int:
        """Process a single file into chunks.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Number of chunks created
        """
        try:
            logger.info(f"Processing {file_path.name}")
            out_dir = self._create_output_dir(file_path)
            
            # Process based on file type
            if file_path.suffix == '.json':
                chunks = self._process_json_file(file_path)
            else:  # .md files
                chunks = self._process_markdown_file(file_path)
            
            # Save chunks
            chunk_count = self._save_chunks(chunks, out_dir)
            logger.info(f"Created {chunk_count} chunks in {out_dir}")
            return chunk_count
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            return 0
    
    def process_all_files(self) -> int:
        """Process all files in the corpus directory.
        
        Returns:
            Total number of chunks created
        """
        total_chunks = 0
        files = list(self.corpus_dir.glob("*.[mj][ds]*"))  # matches both .md and .json
        
        for file_path in tqdm(files, desc="Processing files"):
            total_chunks += self.process_file(file_path)
        
        logger.info(f"Total chunks created: {total_chunks}")
        return total_chunks

def main():
    """Main entry point for the script."""
    try:
        config = ChunkingConfig()
        chunker = DocumentChunker(config)
        total_chunks = chunker.process_all_files()
        logger.info(f"Successfully created {total_chunks} chunks")
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()