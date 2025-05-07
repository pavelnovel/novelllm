from langchain.tools import tool
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import query_chunks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from query_chunks import query_engine

@tool
def query_knowledge_base(topic: str) -> str:
    """Run a semantic query against the embedded corpus.
    
    Args:
        topic: The topic to search for in the knowledge base
        
    Returns:
        A string containing the search results or an error message
    """
    try:
        logger.info(f"Querying knowledge base for topic: {topic}")
        response = query_engine.query(topic)
        result = response.response if hasattr(response, 'response') else str(response)
        logger.info(f"Query successful, response length: {len(result)}")
        return result
    except Exception as e:
        error_msg = f"I encountered an error while searching the knowledge base: {str(e)}"
        logger.error(error_msg)
        return error_msg
