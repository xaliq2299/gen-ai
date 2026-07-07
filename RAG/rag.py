from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from store_in_faiss import load_chunks


# Load FAISS index
def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    return index

# Retrieve relevant chunks
def retrieve_chunks(query, index, text_chunks, k=2):
    """Retrieve the most relevant text chunks for a given query."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)

    # Map indices to the original text chunks
    retrieved_chunks = [text_chunks[i] for i in indices[0]]
    return retrieved_chunks

# Load local LLM
def load_local_llm(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate answer using RAG
def generate_answer(query, retrieved_chunks, llm_pipeline):
    context = " ".join(retrieved_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm_pipeline(prompt, max_new_tokens=100, num_return_sequences=1)
    return response[0]['generated_text']

def retrieve_and_generate(query):
    # Paths
    faiss_index_path = "faiss_index.bin"
    chunked_data_path = "chunked_data.json"

    # Load FAISS index and embeddings
    index = load_faiss_index(faiss_index_path)
    # Load chunks
    indices, chunks = load_chunks(chunked_data_path)

    if not chunks:
        print("No chunks to process.")
        return

    # Load local LLM
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Replace with your local model name if different
    llm_pipeline = load_local_llm("gpt2")

    # Query
    # query = "What does rap mean?"
    # retrieved_chunks = retrieve_chunks(query, index, chunks)
    retrieved_chunks = [""]
    answer = generate_answer(query, retrieved_chunks, llm_pipeline)

    return answer

def main():
    query = "What does rap mean?"
    answer = retrieve_and_generate(query)

    print("Answer:", answer)

if __name__ == "__main__":
    main()
