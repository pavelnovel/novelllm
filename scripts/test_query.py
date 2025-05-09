from query_chunks import VectorStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize vector store
    vector_store = VectorStore()
    
    # Test queries
    test_queries = [
        "What are some key marketing strategies?",
        "How to build a personal brand?",
        "What are the best practices for LinkedIn posts?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.query(query, n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Content: {result['content'][:200]}...")
            print(f"Source: {result['metadata']['file_path']}")
            print(f"Score: {result['score']:.3f}")

if __name__ == "__main__":
    main() 