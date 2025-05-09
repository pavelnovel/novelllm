from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import chromadb
from typing import Optional, List, Dict, Any
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Knowledge base manager for handling document queries."""
    
    def __init__(
        self,
        collection_name: str = "linkedin_chunks",
        store_path: str = "./chroma_store",
        model_name: str = "gpt-4.1-nano",
        top_k: int = 5
    ):
        """Initialize the knowledge base.
        
        Args:
            collection_name: Name of the ChromaDB collection
            store_path: Path to the ChromaDB store
            model_name: Name of the OpenAI model to use
            top_k: Number of chunks to retrieve
        """
        self.collection_name = collection_name
        self.store_path = store_path
        self.model_name = model_name
        self.top_k = top_k
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize the knowledge base components."""
        try:
            # Connect to ChromaDB
            logger.info(f"Connecting to ChromaDB at {self.store_path}")
            self.client = chromadb.PersistentClient(path=self.store_path)
            self.collection = self.client.get_collection(self.collection_name)
            
            # Create vector store
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            
            # Configure LLM
            Settings.llm = OpenAI(model_name=self.model_name)
            
            # Create index
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
            
            # Create retriever with more chunks
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.top_k,
                filters=None  # Remove any filters that might be limiting results
            )
            
            # Create response synthesizer with more detailed mode
            self.response_synthesizer = get_response_synthesizer(
                response_mode="tree_summarize",  # Changed from "compact" to get more detailed responses
                structured_answer_filtering=False  # Don't filter out any content
            )
            
            # Create query engine with more detailed configuration
            self.query_engine = RetrieverQueryEngine(
                retriever=self.retriever,
                response_synthesizer=self.response_synthesizer,
                node_postprocessors=None  # Remove any postprocessors that might filter results
            )
            
            logger.info("Knowledge base initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            raise
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """Execute a query against the knowledge base."""
        try:
            logger.info(f"Executing query: {query_text}")
            
            # Execute query - remove the incorrect parameters
            response = self.query_engine.query(query_text)
            
            # Extract source information
            sources = []
            for source_node in response.source_nodes:
                if hasattr(source_node, 'node'):
                    source_info = {
                        'score': source_node.score,
                        'file_path': source_node.node.metadata.get('file_path', 'Unknown'),
                        'content': source_node.node.text[:200] + '...' if len(source_node.node.text) > 200 else source_node.node.text
                    }
                    sources.append(source_info)
            
            # Format the complete response with sources
            formatted_response = []
            
            # Add the main response
            formatted_response.append("Response:")
            formatted_response.append("---------")
            formatted_response.append(response.response)
            formatted_response.append("")
            
            # Add sources section
            if sources:
                formatted_response.append("Sources:")
                formatted_response.append("--------")
                for source in sources:
                    formatted_response.append(f"📑 {source['file_path']} (Score: {source['score']:.4f})")
                    formatted_response.append(f"   Preview: {source['content']}")
                    formatted_response.append("")
            
            # Join everything together
            complete_response = "\n".join(formatted_response)
            
            # Return both the formatted response and raw sources
            result = {
                'response': complete_response,
                'sources': sources
            }
            
            logger.info(f"Query completed successfully with {len(sources)} sources")
            return result
            
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            logger.error(error_msg)
            return {
                'response': error_msg,
                'sources': []
            }

# Create a singleton instance
knowledge_base = KnowledgeBase()

# Expose the query engine for direct use
query_engine = knowledge_base.query_engine

if __name__ == "__main__":
    # Get query from command line argument or prompt user
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
    else:
        query_text = input("Enter your query: ")

    # Execute the query
    result = knowledge_base.query(query_text)

    # Print the response
    print("\n=== RESPONSE ===")
    print(result['response'])

    # Print the source documents and their scores
    print("\n=== SOURCES ===")
    for source in result['sources']:
        print(f"\nScore: {source['score']:.4f}")
        print(f"File: {source['file_path']}")
        print(f"Preview: {source['content']}")
