from pathlib import Path
import os, chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import logging
from tqdm import tqdm  # For progress indication

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure environment variable is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize ChromaDB
logger.info("Initializing ChromaDB client")
client = chromadb.PersistentClient(path="chroma_store")
collection = client.get_or_create_collection("linkedin_chunks")
embedder = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))

# Batch processing constants
BATCH_SIZE = 100  # Adjust based on your OpenAI rate limits and token needs

# Collect files to process
chunk_dirs = list(Path("chunks").iterdir())
logger.info(f"Found {len(chunk_dirs)} directories to process")

# Process files in batches
texts, ids = [], []
total_chunks = 0

for dir_path in tqdm(chunk_dirs, desc="Processing directories"):
    dir_files = list(dir_path.glob("*.txt"))
    logger.info(f"Processing {dir_path.name}: {len(dir_files)} files")
    
    for f in dir_files:
        try:
            txt = f.read_text(encoding="utf-8").strip()
            if txt:
                texts.append(txt)
                ids.append(f"{dir_path.name}_{f.stem}")
                
                # Process in batches to avoid hitting API limits
                if len(texts) >= BATCH_SIZE:
                    logger.info(f"Embedding batch of {len(texts)} chunks")
                    vectors = embedder(texts)  # OpenAIEmbeddingFunction is callable
                    collection.add(documents=texts, embeddings=vectors, ids=ids)
                    total_chunks += len(texts)
                    texts, ids = [], []  # Reset for next batch
        except Exception as e:
            logger.error(f"Error processing {f}: {e}")

# Process any remaining texts
if texts:
    logger.info(f"Embedding final batch of {len(texts)} chunks")
    vectors = embedder(texts)  # OpenAIEmbeddingFunction is callable
    collection.add(documents=texts, embeddings=vectors, ids=ids)
    total_chunks += len(texts)
