from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
import chromadb
from typing import Optional
import sys

# Connect to the persistent ChromaDB
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_collection("linkedin_chunks")

# Create a vector store from the Chroma collection
vector_store = ChromaVectorStore(chroma_collection=collection)

# Configure the LLM using Settings
Settings.llm = OpenAI(model_name="gpt-4.1-nano")  # Using GPT-4.1 Nano for cost efficiency

# Create an index from the vector store
index = VectorStoreIndex.from_vector_store(vector_store)

# Create a query engine
query_engine = index.as_query_engine(
    similarity_top_k=5,  # Number of chunks to retrieve
)

if __name__ == "__main__":
    # Get query from command line argument or prompt user
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
    else:
        query_text = input("Enter your query: ")

    # Execute the query
    response = query_engine.query(query_text)

    # Print the response
    print("\n=== RESPONSE ===")
    print(response.response)

    # Print the source documents and their scores
    print("\n=== SOURCES ===")
    for source_node in response.source_nodes:
        print(f"Score: {source_node.score:.4f} | {source_node.node.metadata['file_path']}")
