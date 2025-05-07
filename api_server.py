from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from query_chunks import query_engine
from tools.query_tool import query_knowledge_base

app = FastAPI(
    title="NovelLLM API",
    description="API for querying your knowledge base using LlamaIndex and LangChain",
    version="0.1.0"
)

class QueryResponse(BaseModel):
    response: str
    sources: List[dict]

@app.get("/")
def read_root():
    return {"status": "online", "info": "NovelLLM API is running"}

@app.get("/draft", response_model=QueryResponse)
def get_draft(topic: str = Query(..., description="Topic to generate content about")):
    """Generate content based on the provided topic."""
    # Query the knowledge base
    response = query_engine.query(topic)
    
    # Format the sources
    sources = []
    for source_node in response.source_nodes:
        sources.append({
            "score": round(source_node.score, 4),
            "file_path": source_node.node.metadata.get('file_path', 'Unknown')
        })
    
    return QueryResponse(
        response=response.response,
        sources=sources
    )

@app.get("/agent")
def agent_query(query: str = Query(..., description="Query for the LangChain agent")):
    """Execute a query using the LangChain agent."""
    result = query_knowledge_base(query)
    return {"response": result}

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
