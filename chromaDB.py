import re                   # Text preprocessing
import string               # For punctuation
import nltk                 # Tokenization and stopwords
import ollama               # For LLM and embeddings via Ollama or custom models
import numpy as np          # Array operations
import fitz                 # PDF text extraction
from tqdm import tqdm       # Progress bar
import os                   # File system navigation
import time                 # Timing
import tracemalloc          # Memory usage tracking
from sentence_transformers import SentenceTransformer  # Embedding model
from collections import Counter     # For simple frequency counting

import chromadb

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Constants and configuration
VECTOR_DIM = 768
COLLECTION_NAME = "slides" 
QUERY = 'What is a Binary Tree?'
LLM_MODEL = 'llama3.2:latest'
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'

# 
os.environ["CHROMA_API_URL"] = "http://localhost:8000"

# Initialize Chroma client
client = chromadb.Client()

# Create a new collection for documents
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    collection.delete()  # Clear out existing documents if needed
    collection = client.create_collection(name=COLLECTION_NAME)
except Exception:
    collection = client.create_collection(name=COLLECTION_NAME)

# Initialize the embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Function to generate an embedding for a given text
def get_embedding(text: str) -> list:
    # Generate the embedding and convert it to a list of floats
    response = embedding_model.encode(text)
    return response.tolist()

# Extract text from each page of a PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

# Preprocess text: remove newlines, tokenize, and handle symbols
def preprocess_text(text):
    text = text.replace('\n', ' ').strip()
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = ["<SYM>" if re.fullmatch(r"[^\w\d" + re.escape(string.punctuation) + "]", token) else token for token in tokens]
    return tokens

# Split text into chunks of approximately `chunk_size` words with an overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = preprocess_text(text)
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

# Store a chunk and its embedding into the Chroma collection
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    # Generate a document ID (here using a hash of the chunk for uniqueness)
    doc_id = f"{file}_page_{page}_chunk_{hash(chunk)}"
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}],
        ids=[doc_id]
    )
    # Uncomment to debug:
    # print(f"Stored embedding for: {chunk}")

# Process all PDFs in the specified directory
def process_pdfs(data_dir):
    tracemalloc.start()
    start_time = time.time()
    for file_name in tqdm(os.listdir(data_dir)):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk in chunks:
                    embedding = get_embedding(chunk)
                    store_embedding(file=file_name, page=str(page_num), chunk=chunk, embedding=embedding)
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f'Time elapsed: {round(elapsed, 4)} seconds')
    print(f"Peak memory usage: {peak / 1024**2:.2f} MiB")

# Query the Chroma collection using a query text
def query_chroma(query_text: str, n_results=5):
    embedding = get_embedding(query_text)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )
    for i in range(len(results['ids'])):
        print(f"ID: {results['ids'][i]}")
        print(f"Document: {results['documents'][i]}")
        print(f"Metadata: {results['metadatas'][i]}")
        if 'distances' in results:
            print(f"Distance: {results['distances'][i]}")
        print("-----")
    return results

# A helper function to wrap the query result in a simplified format
def search_embeddings(query, top_k=3):
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    
    top_results = []
    
    # Extract the nested lists for metadata and distances
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


# Generate a response using RAG
def generate_rag_response(query, context_results):
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

# --- Main execution ---
print("Processing PDFs...")
process_pdfs("Slides/")  # Adjust the directory to point to your PDF files
print("\n---Done processing PDFs---\n")

print("Querying collection with query:", QUERY)
results = query_chroma(QUERY)

print("RAG Response:")
rag_response = generate_rag_response(QUERY, search_embeddings(QUERY))
print(rag_response)