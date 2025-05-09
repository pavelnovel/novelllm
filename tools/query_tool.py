from langchain.tools import tool
from langchain_core.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import sys
import os
import logging
from langchain_core.callbacks import CallbackManagerForToolRun

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import query_chunks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from query_chunks import knowledge_base

class QueryKnowledgeBaseInput(BaseModel):
    """Input for the query knowledge base tool."""
    topic: str = Field(..., description="The topic to search for in the knowledge base")

class QueryKnowledgeBaseTool(BaseTool):
    """Tool for querying the knowledge base."""
    name: str = "query_knowledge_base"
    description: str = "Run a semantic query against the embedded corpus."
    args_schema: Type[BaseModel] = QueryKnowledgeBaseInput
    
    def _run(
        self,
        topic: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query against the knowledge base."""
        try:
            logger.info(f"Querying knowledge base for topic: {topic}")
            result = knowledge_base.query(topic)
            
            # Return the complete formatted response
            logger.info(f"Query successful, response length: {len(result['response'])}")
            return result['response']
            
        except Exception as e:
            error_msg = f"I encountered an error while searching the knowledge base: {str(e)}"
            logger.error(error_msg)
            return error_msg

# Create the tool instance
query_knowledge_base = QueryKnowledgeBaseTool()
