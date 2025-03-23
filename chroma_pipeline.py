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
import time

import chromadb
import os
import time
import tracemalloc
import numpy as np
import fitz  # PyMuPDF for PDF processing
import ollama
import nltk
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Constants and configuration
VECTOR_DIM = 768
COLLECTION_NAME = "slides" 
QUERY = 'What is a AVL Tree?'
COLLECTION_NAME = "slides"
LLM_MODEL = 'llama3.2:latest'
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
DATA_DIR = "Slides/"  # Adjust this to your directory containing PDFs

# Initialize Qdrant client
qdrant_client = QdrantClient("http://localhost:6333")

# Create or reset Qdrant collection
qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
)

# Initialize embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Function to generate an embedding for a given text
def get_embedding(text: str) -> list:
    response = embedding_model.encode(text)
    return response.tolist()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


def preprocess_text(text):
    """Preprocess text by tokenizing and handling symbols."""
    text = text.replace('\n', ' ').strip()
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens


def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks with overlap."""
    words = preprocess_text(text)
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def get_embedding(text: str) -> list:
    """Generate an embedding for a given text."""
    return embedding_model.encode(text).tolist()


def store_embedding(file: str, page: str, chunk: str, embedding: list):
    """Store chunk embedding in Qdrant."""
    doc_id = hash(f"{file}_page_{page}_chunk_{chunk}")

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=doc_id, vector=embedding, payload={"file": file, "page": page, "chunk": chunk})
        ],
    )
    # Uncomment for debugging:
    # print(f"Stored embedding for: {chunk}")


def process_pdfs(data_dir):
    """Process PDFs, extract text, generate embeddings, and store in Qdrant."""
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


def query_qdrant(query_text: str, n_results=5):
    """Retrieve top-k similar results from Qdrant."""
    query_embedding = get_embedding(query_text)

    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=n_results,
    )

    top_results = []
    for result in results:
        meta = result.payload
        top_results.append({
            "file": meta.get("file", "Unknown file"),
            "page": meta.get("page", "Unknown page"),
            "chunk": meta.get("chunk", "Unknown chunk"),
            "similarity": result.score,
        })

    return top_results


def generate_rag_response(query, context_results):
    """Generate a response using Ollama and retrieved context from Qdrant."""
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


def run_test(queries, embedding_model, llm_model, chunk_size=300, overlap=50):
    global collection

    try:
        collection.delete()
    except Exception as e:
        print("Error deleting collection:", e)

    collection = client.create_collection(name=COLLECTION_NAME)

    print('Processing PDFs...')
    process_pdfs("Slides/")
    print("\n---Done processing PDFs---\n")

def run_qdrant_pipeline(query_text):
    """Complete Qdrant-based RAG pipeline: PDF processing → Query → Response."""
    print("Processing PDFs...")
    process_pdfs(DATA_DIR)  
    print("\n---Done processing PDFs---\n")

    for query in queries:
        print('Query:', query)
        start_time = time.time()
        print(generate_rag_response(query, search_embeddings(query)))

        elapsed = time.time() - start_time
        print(f'Time elapsed: {round(elapsed, 4)} seconds')
        print('---------------------------')

    print(f"Querying Qdrant with query: {query_text}")
    search_results = query_qdrant(query_text)

    print("\nRAG Response:")
    rag_response = generate_rag_response(query_text, search_results)
    print(rag_response)

# --- Run Pipeline ---
# if __name__ == "__main__":
#     user_query = input("\nEnter your query: ")
#     run_qdrant_pipeline(user_query)
