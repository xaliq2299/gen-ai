import faiss
from sentence_transformers import SentenceTransformer
import os, json


def load_chunks(file_path):
    """Load chunks and their indices from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        # Return both indices and text values
        return list(data.keys()), list(data.values())
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return [], []
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []

def generate_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    """Generate vector embeddings for a list of text chunks."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

def store_in_faiss(embeddings, indices, index_path):
    """Store embeddings in a FAISS index."""
    if embeddings is None or len(embeddings) == 0:
        print("No embeddings to store in FAISS index.")
        return

    # Create a FAISS index
    embedding_dimension = embeddings.shape[1]
    base_index = faiss.IndexFlatL2(embedding_dimension) # L2 distance
    index = faiss.IndexIDMap(base_index) # Use IDMap to store custom indices

    # Add embeddings to the index
    index.add_with_ids(embeddings, indices)
    print(f"Number of embeddings in the index: {index.ntotal}")

    # Save the index to a file
    try:
        faiss.write_index(index, index_path)
        print(f"FAISS index saved to {index_path}")
    except Exception as e:
        print(f"An error occurred while saving the FAISS index: {e}")


def main():
    # Define file paths
    chunks_file_path = os.path.join(os.path.dirname(__file__), 'chunked_data.json')
    faiss_index_path = os.path.join(os.path.dirname(__file__), 'faiss_index.bin')

    # Load chunks
    indices, chunks = load_chunks(chunks_file_path)

    if not chunks:
        print("No chunks to process.")
        return

    # Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Store embeddings in FAISS index
    store_in_faiss(embeddings, indices, faiss_index_path)

if __name__ == "__main__":
    main()
