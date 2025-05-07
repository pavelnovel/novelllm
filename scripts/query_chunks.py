import os
import logging
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_retrieval_engine():
    try:
        # Configure LLM
        logger.info("Configuring LLM")
        llm = OpenAI(model_name="gpt-4.1-nano")
        service_context = ServiceContext.from_defaults(llm=llm)
        
        # Connect to Chroma DB
        logger.info("Connecting to ChromaDB")
        chroma_client = chromadb.PersistentClient(path="./chroma_store")
        collection = chroma_client.get_or_create_collection("linkedin_chunks")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index from vector store
        logger.info("Creating index from vector store")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            service_context=service_context
        )
        
        # Configure query engine with parameters
        logger.info("Configuring query engine")
        query_engine = index.as_query_engine(
            similarity_top_k=5,  # Number of chunks to retrieve
            response_mode="compact"  # Options: "default", "compact", "tree_summarize"
        )
        
        return query_engine
        
    except Exception as e:
        logger.error(f"Error setting up retrieval engine: {e}")
        raise

def run_query(query_text):
    try:
        logger.info(f"Running query: {query_text}")
        query_engine = setup_retrieval_engine()
        response = query_engine.query(query_text)
        return response
    except Exception as e:
        logger.error(f"Error running query: {e}")
        raise

if __name__ == "__main__":
    # Example query
    query = "write a post about hiring"
    response = run_query(query)
    
    print("\n" + "="*50)
    print("QUERY RESPONSE:")
    print("="*50)
    print(response)
    print("="*50)
    
    # Optional: Print source nodes/documents that were used
    print("\nSource documents used:")
    for node in response.source_nodes:
        print(f"- Score: {node.score:.4f}")
        print(f"- Source: {node.node.metadata}")
        print(f"- Text snippet: {node.node.text[:150]}...")
        print("-"*30)