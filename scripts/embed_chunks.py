from pathlib import Path
import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import logging
from tqdm import tqdm
from typing import List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for the embedding process."""
    batch_size: int = 100
    collection_name: str = "linkedin_chunks"
    store_path: str = "chroma_store"
    chunk_dir: str = "chunks"

class DocumentProcessor:
    """Handles document processing and embedding."""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize the document processor.
        
        Args:
            config: Configuration for the embedding process
        """
        self.config = config
        self._validate_environment()
        self._init_components()
    
    def _validate_environment(self):
        """Validate the environment setup."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    def _init_components(self):
        """Initialize ChromaDB and embedding components."""
        try:
            logger.info("Initializing ChromaDB client")
            self.client = chromadb.PersistentClient(path=self.config.store_path)
            self.collection = self.client.get_or_create_collection(
                self.config.collection_name
            )
            self.embedder = OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info("Components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def _process_file(self, file_path: Path) -> Tuple[str, str]:
        """Process a single file and return its content and ID.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Tuple of (text content, document ID)
        """
        try:
            text = file_path.read_text(encoding="utf-8").strip()
            if text:
                return text, f"{file_path.parent.name}_{file_path.stem}"
            return "", ""
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return "", ""
    
    def _process_batch(self, texts: List[str], ids: List[str]) -> int:
        """Process a batch of texts.
        
        Args:
            texts: List of text contents
            ids: List of document IDs
            
        Returns:
            Number of successfully processed documents
        """
        try:
            if not texts:
                return 0
                
            logger.info(f"Embedding batch of {len(texts)} chunks")
            vectors = self.embedder(texts)
            
            # Create metadata for each document
            metadatas = []
            for doc_id in ids:
                # Extract file path from ID (format: directory_filename)
                parts = doc_id.split('_', 1)
                if len(parts) == 2:
                    directory, filename = parts
                    file_path = f"{directory}/{filename}"
                else:
                    file_path = doc_id
                
                metadatas.append({
                    "file_path": file_path,
                    "source": "linkedin_posts"
                })
            
            # Add documents with metadata
            self.collection.add(
                documents=texts,
                embeddings=vectors,
                ids=ids,
                metadatas=metadatas
            )
            return len(texts)
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return 0
    
    def process_documents(self) -> int:
        """Process all documents in the chunks directory.
        
        Returns:
            Total number of processed chunks
        """
        chunk_dirs = list(Path(self.config.chunk_dir).iterdir())
        logger.info(f"Found {len(chunk_dirs)} directories to process")
        
        total_chunks = 0
        texts, ids = [], []
        
        for dir_path in tqdm(chunk_dirs, desc="Processing directories"):
            dir_files = list(dir_path.glob("*.txt"))
            logger.info(f"Processing {dir_path.name}: {len(dir_files)} files")
            
            for file_path in dir_files:
                text, doc_id = self._process_file(file_path)
                if text and doc_id:
                    texts.append(text)
                    ids.append(doc_id)
                    
                    if len(texts) >= self.config.batch_size:
                        total_chunks += self._process_batch(texts, ids)
                        texts, ids = [], []
        
        # Process remaining texts
        if texts:
            total_chunks += self._process_batch(texts, ids)
        
        logger.info(f"Completed processing {total_chunks} chunks")
        return total_chunks

def main():
    """Main entry point for the script."""
    try:
        config = EmbeddingConfig()
        processor = DocumentProcessor(config)
        total_chunks = processor.process_documents()
        logger.info(f"Successfully processed {total_chunks} chunks")
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
