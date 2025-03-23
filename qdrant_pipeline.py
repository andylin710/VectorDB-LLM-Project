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

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Important constants
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
COLLECTION_NAME = "slides"
DISTANCE_METRIC = "COSINE"
QUERY = 'What is the CAP Theorem?'
LLM_MODEL = 'llama3.2:latest'
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'

# Initialize Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333")


# Create or reset Qdrant collection
qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
)

# Initialize embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def get_embedding(text: str, embedding_model) -> list:
    
    model = SentenceTransformer(embedding_model)
    response = model.encode(text)
    return response

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
    """Preprocess text by tokenizing and removing unnecessary symbols."""
    text = text.replace('\n', ' ').strip()
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens

# Split text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately `chunk_size` words with overlap."""
    words = preprocess_text(text)
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

# Store embeddings in Qdrant
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    """Store chunk embedding in Qdrant."""
    doc_id = abs(hash(f"{file}_page_{page}_chunk_{chunk}")) 


    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=doc_id, vector=embedding, payload={"file": file, "page": page, "chunk": chunk})
        ],
    )
    # Uncomment for debugging:
    # print(f"Stored embedding for: {chunk}")

# Process all PDF files in a given directory
def process_pdfs(data_dir, model, chunk_size=300, overlap=50):

    tracemalloc.start()
    start_time = time.time()

    for file_name in tqdm(os.listdir(data_dir)):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, model)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),
                        embedding=embedding,
                    )

    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f'Time elapsed: {round(elapsed, 4)} seconds')
    print(f"Peak memory usage: {peak / 1024**2:.2f} MiB")

# Query Qdrant for similar documents
def query_qdrant(query_text: str, model, n_results=5):
    """Retrieve top-k similar results from Qdrant."""
    query_embedding = get_embedding(query_text, model)

    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=n_results,
    )

    for i, result in enumerate(results):
        print(f"ID: {result.id}")
        print(f"Chunk: {result.payload['chunk']}")
        print(f"File: {result.payload['file']}")
        print(f"Page: {result.payload['page']}")
        print(f"Similarity Score: {result.score:.4f}")
        print("-----")

    return results

# Search function that formats results nicely
def search_embeddings(query, model, top_k=3):
    """Search embeddings in Qdrant and return formatted results."""
    embedding = get_embedding(query, model)
    
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=top_k
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

# Generate a response using RAG
def generate_rag_response(query, context_results, model):
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
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def clear_qdrant_store():
    """Clears the Qdrant vector store by deleting and recreating the collection."""
    print("Clearing existing Qdrant store...")

    # Delete the existing collection
    try:
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        print("Qdrant collection deleted successfully.")
    except Exception as e:
        print(f"Error deleting Qdrant collection: {e}")

    # Recreate the collection with the same vector configuration
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    
    print("Qdrant store cleared and collection recreated.")

def create_qdrant_index():
    """
    Ensures Qdrant collection exists with the correct HNSW-like configuration.
    
    If the collection already exists, it is deleted and recreated to prevent issues.
    """
    print("Setting up Qdrant collection...")

    # Delete the existing collection if it exists
    try:
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        print("Existing Qdrant collection deleted.")
    except Exception as e:
        print(f"Warning: {e}")

    # Recreate the collection with the correct indexing strategy
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    print("Qdrant collection created successfully.")


def run_test(queries, embedding_model, llm_model, chunk_size=300, overlap=50):
    qdrant_client = QdrantClient(url="http://localhost:6333")

    """Runs a full test of the Qdrant pipeline with queries, embedding model, and LLM."""
    
    clear_qdrant_store()
    create_qdrant_index()

    print("Running Qdrant Pipeline Test...")
    
    print('Processing PDFs...')
    process_pdfs("Slides/", embedding_model, chunk_size, overlap)
    print("\n---Done processing PDFs---\n")

    for query in queries:
        print(f'Query: {query}')
        start_time = time.time()

        # Perform query in Qdrant
        response = generate_rag_response(query, search_embeddings(query, embedding_model), llm_model)

        elapsed = time.time() - start_time
        print(response)
        print(f'Time elapsed: {round(elapsed, 4)} seconds')
        print('---------------------------')