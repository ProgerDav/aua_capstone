from time import time
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("ACTIVELOOP_TOKEN")

# Default API endpoint
DEFAULT_API_PATH = "https://api.activeloop.dev"


def retrieve_from_vector_db(query: str, top_k: int = 15, token: str = None, model_id: str = "activeloop-l0-retrieval", base_url: str = DEFAULT_API_PATH):
    """
    Retrieve relevant information from vector database using the activeloop-l0-retrieval model.
    """
    # Use token if provided, otherwise use environment variable
    current_token = token if token else api_key
    
    try:
        t0 = time()
        
        # Initialize client
        client = OpenAI(
            api_key=current_token,
            base_url=base_url,
        )
        
        # Make the request (non-streaming)
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": query}],
            stream=False,
        )
        
        # Extract relevant docs from response
        relevant_docs = [
            {**c.metadata, "text": c.message.content} for c in response.choices
        ]
        
        t1 = time()
        print(f"Time taken: {t1 - t0} seconds")
        
        return relevant_docs
    except Exception as e:
        print(f"Error retrieving from vector DB: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    results = retrieve_from_vector_db("What is machine learning?")
    print(f"Found {len(results)} results")


