import os
import logging
import chromadb
from typing import List, Dict, Any
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from embed_chunks import EmbeddingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        """Initialize the vector store."""
        try:
            # Initialize configuration
            self.config = EmbeddingConfig()
            
            # Initialize embedder
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.embedder = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Connect to ChromaDB
            logger.info("Connecting to ChromaDB")
            self.client = chromadb.PersistentClient(path=self.config.store_path)
            self.collection = self.client.get_or_create_collection(self.config.collection_name)
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def query(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the vector store.
        
        Args:
            query: Query string
            n_results: Number of results to return
            
        Returns:
            List of dictionaries containing results
        """
        try:
            # Get query embedding
            query_vector = self.embedder([query])[0]
            
            # Query ChromaDB directly
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] else {"file_path": "unknown"}
                distance = results["distances"][0][i]
                
                # Extract file path from content if metadata is missing
                if "file_path" not in metadata:
                    try:
                        import json
                        content_json = json.loads(content)
                        file_path = f"consolidated/linkedin_posts_chunk_{content_json.get('chunk_id', 'unknown')}"
                        metadata["file_path"] = file_path
                    except:
                        metadata["file_path"] = "unknown"
                
                formatted_results.append({
                    "content": content,
                    "metadata": metadata,
                    "score": 1 - distance  # Convert distance to similarity score
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            return []

if __name__ == "__main__":
    # Example query
    vector_store = VectorStore()
    query = "write a post about hiring"
    results = vector_store.query(query)
    
    print("\n" + "="*50)
    print("QUERY RESULTS:")
    print("="*50)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {result['content'][:200]}...")
        print(f"Source: {result['metadata']['file_path']}")
        print(f"Score: {result['score']:.3f}")
        print("-"*30)