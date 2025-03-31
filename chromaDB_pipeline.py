import re                   # Text preprocessing
import string               # For punctuation
import nltk                 # Tokenization and stopwords
import ollama               # LLM calls
import numpy as np          # Array operations
import fitz                 # PDF text extraction (PyMuPDF)
from tqdm import tqdm       # Progress bar
import os                   # File system navigation
import time                 # Timing
import tracemalloc          # Memory usage tracking
import csv                  # For CSV writing
import cpuinfo              # CPU info
import psutil               # RAM info

from sentence_transformers import SentenceTransformer  # Embedding model
from collections import Counter

import chromadb

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# === Constants and Configuration ===
VECTOR_DIM = 768
COLLECTION_NAME = "slides" 
DATA_DIR = "Slides/"  # Directory containing your PDFs
LLM_MODEL = 'llama3.2:latest'
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'

# Set the Chroma API URL and initialize client and collection
os.environ["CHROMA_API_URL"] = "http://localhost:8000"
client = chromadb.Client()
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    collection.delete()  # Clear out existing documents if needed
    collection = client.create_collection(name=COLLECTION_NAME)
except Exception:
    collection = client.create_collection(name=COLLECTION_NAME)

# Initialize the embedding model globally.
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# === Helper Functions ===

def get_embedding(text: str) -> list:
    """
    Generate an embedding for a given text and return it as a list of floats.
    """
    response = embedding_model.encode(text)
    return response.tolist()

def extract_text_from_pdf(pdf_path):
    """
    Extract text by page from a PDF.
    Returns a list of tuples: (page_number, page_text).
    """
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

def preprocess_text(text):
    """
    Preprocess the text by replacing newlines and tokenizing.
    Symbols that do not match word/digit or punctuation are replaced with <SYM>.
    """
    text = text.replace('\n', ' ').strip()
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [
        "<SYM>" if re.fullmatch(r"[^\w\d" + re.escape(string.punctuation) + "]", token)
        else token
        for token in tokens
    ]
    return tokens

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """
    Split text into chunks of approximately chunk_size words with overlap.
    """
    words = preprocess_text(text)
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def store_embedding(file: str, page: str, chunk: str, embedding: list):
    """
    Store the chunk and its embedding in the Chroma collection.
    A unique document ID is generated from the file, page, and a hash of the chunk.
    """
    doc_id = f"{file}_page_{page}_chunk_{hash(chunk)}"
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}],
        ids=[doc_id]
    )

def process_pdfs(data_dir, chunk_size=300, overlap=50):
    """
    Process all PDFs in the specified directory:
      - Extract text from each page,
      - Split the text into overlapping chunks,
      - Generate embeddings and store them in Chroma.
    Returns the elapsed time and peak memory usage.
    """
    tracemalloc.start()
    start_time = time.time()
    for file_name in tqdm(os.listdir(data_dir)):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                for chunk in chunks:
                    embedding = get_embedding(chunk)
                    store_embedding(file=file_name, page=str(page_num), chunk=chunk, embedding=embedding)
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f'Time elapsed for indexing: {round(elapsed, 4)} seconds')
    print(f"Peak memory usage during indexing: {peak / 1024**2:.2f} MiB")
    return round(elapsed, 4), round(peak / 1024**2, 2)

def search_embeddings(query, top_k=3):
    """
    Search the Chroma collection for documents similar to the query.
    Returns a list of top results with file, page, chunk, and similarity.
    """
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    
    top_results = []
    # Extract metadata and distances (if available)
    metadata_list = results["metadatas"][0]
    distances = results["distances"][0] if "distances" in results else [None] * len(metadata_list)
    
    for i in range(len(results["ids"][0])):
        meta = metadata_list[i]
        top_results.append({
            "file": meta.get("file", "Unknown file"),
            "page": meta.get("page", "Unknown page"),
            "chunk": meta.get("chunk", "Unknown chunk"),
            "similarity": distances[i],
        })
    
    return top_results

def generate_rag_response(query, context_results):
    """
    Generate a RAG (Retrieval-Augmented Generation) response.
    The context is built from the metadata of the search results.
    Uses Ollama to generate the answer.
    """
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )
    prompt = f"""You are a helpful AI assistant. 
Use the following context to answer the query as accurately as possible. If the context is 
not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""
    response = ollama.chat(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def get_cpu_type():
    """Return the CPU brand string."""
    return cpuinfo.get_cpu_info()['brand_raw']

def get_ram_size():
    """Return the total RAM size in GiB."""
    return round(psutil.virtual_memory().total / (1024 ** 3))

# === Main Pipeline Function (run_test) ===

def run_test(queries, embedding_model_name, llm_model, chunk_size=300, overlap=50):
    """
    Clears the existing Chroma collection, reindexes PDFs, then processes each query:
      - For each query, it retrieves similar documents,
      - Generates a RAG response,
      - Measures query time,
      - Logs performance (CPU type, RAM size, index time/memory, query time, etc.) to a CSV,
      - And saves all responses to a text file.
    """
    global collection, embedding_model

    # Delete and recreate the Chroma collection
    try:
        client.delete_collection(name=COLLECTION_NAME)

    except Exception as e:
        print("Error deleting collection:", e)
    collection = client.create_collection(name=COLLECTION_NAME)

    # Reinitialize the embedding model (if needed)
    embedding_model = SentenceTransformer(embedding_model_name)

    print('Processing PDFs and indexing embeddings...')
    index_elapsed, index_memory = process_pdfs(DATA_DIR, chunk_size, overlap)
    print("\n---Done processing PDFs---\n")

    # CSV file for logging results
    csv_filename = "roland_chroma_test_results.csv"
    answers = []

    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        # Write header only if the file is empty
        if file.tell() == 0:
            writer.writerow([
                "compute_type", "memory_size", "embedding_model", "llm_model", 
                "index_elapsed", "index_memory", "query", "query_time_elapsed",
                "chunk_size", "overlap"
            ])

        for query in queries:
            print('Query:', query)
            start_time = time.time()
            
            # Search embeddings and generate response using the RAG pipeline
            results = search_embeddings(query)
            response = generate_rag_response(query, results)
            print("\nRAG Response:")
            print(response)
            answers.append(response)
            
            elapsed = time.time() - start_time
            print(f'Time elapsed for query: {round(elapsed, 4)} seconds')
            print('---------------------------')
            
            cpu_type = get_cpu_type()
            ram_size = get_ram_size()
            
            writer.writerow([
                cpu_type, ram_size, embedding_model_name, llm_model, 
                index_elapsed, index_memory, query, round(elapsed, 4),
                chunk_size, overlap
            ])

    print(f"Results saved to {csv_filename}")

    answers_text = '\n------------------------\n'.join(answers)
    output_filename = f"QUERY RESULTS_Chroma_{embedding_model_name.split('/')[-1]}_{llm_model.replace('.', '_').replace(':', '_')}_{chunk_size}_{overlap}.txt"
    with open(output_filename, 'w') as file:
        file.write(answers_text)
    print(f"Query responses saved to {output_filename}")

# === Main Entry Point ===

if __name__ == "__main__":
    # Define your test queries here
    test_queries = ['What is an AVL Tree?']
    run_test(test_queries, EMBEDDING_MODEL, LLM_MODEL)
