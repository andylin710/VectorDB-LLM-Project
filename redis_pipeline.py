## DS 4300 Example - from docs
import re                   # Text preprocessing stuff
import string               # More text preprocessing
import nltk                 # Tokenization
import csv                  # CSV writing

import ollama               # Ollama
import redis                # Redis
import numpy as np          # Numpy
import fitz                 # PDF Reader

from tqdm import tqdm       # Progress bar bc I'm impatient
import os                   # Navigate folders
import time                 # Timing
import tracemalloc          # Memory Usage

from sentence_transformers import SentenceTransformer       # Embedding Model
from collections import Counter                             # Simple counting dictionary
from redis.commands.search.query import Query               # Querying 
from redis.commands.search.field import VectorField, TextField

# Libraries for nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Important constants
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "slides"
DISTANCE_METRIC = "COSINE"
QUERY = 'What is the CAP Theorem?'
LLM_MODEL = 'llama3.2:latest'
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

def clear_redis_store():
    """
    Clears redis database store to prevent overlap

    Parameters: 
        None

    Returns:
        None
    """
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")


def create_hnsw_index():
    """
    Clears an HNSW index in the Redis database

    Parameters: 
        None

    Returns:
        None
    """
    # Removes index if it already exists
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    # Creates index
    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


def get_embedding(text: str, embedding_model) -> list:
    """
    Generate an embedding via a specified model. 

    Parameters:
        text (str): Text to embed
        embedding_model (str): Name of the embedding model to use

    Returns:
        response (np.Array): Numerical array representation of the embeddings
    
    """
    model = SentenceTransformer(embedding_model)
    response = model.encode(text)
    return response


def store_embedding(file: str, page: str, chunk: str, embedding: list):
    """
    Stores the embeddings in the Redis index

    Parameters:
        file (str): Name of the file for indexing
        page (str): Page number of the file
        chunk (str): Chunk number of the file
        embedding (list): Embedding representation of a single chunk

    Returns:
        None
    """

    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  
        },
    )


def extract_text_from_pdf(pdf_path):
    """
    Extract text by page from the pdf

    Parameters:
        pdf_path (str): Path to the pdf file

    Returns:
        text_by_page (list): List of the text on each page
    """

    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


def preprocess_text(text):
    """
    Preprocesses and tokenizes text. Steps can be commented out if need be. 

    Parameters:
        text (str): Text to be tokenized

    Returns
        tokens (list): List of tokenized words
    """

    # Replace new lines
    text = text.replace('\n', ' ').strip()

    # Normalize case
    # text = text.lower()

    # Tokenization
    tokens = nltk.tokenize.word_tokenize(text)

    # Remove stopwords if need be
    # tokens = remove_stopwords(tokens)

    # Replaces wacky symbols (like stylized bullets) with <SYM> token if need be
    tokens = ["<SYM>" if re.fullmatch(r"[^\w\d" + re.escape(string.punctuation) + "]", token) else token for token in tokens]

    # Replaces words that show up only once with <UNK> token if need be
    # rare = [item[0] for item in Counter(tokens).items() if item[1] == 1]
    # tokens = ['<UNK>' if token in rare else token for token in tokens]

    # Replaces pure numbers with <NUM> token if need be
    # tokens = ['<NUM>' if token.isdigit() else token for token in tokens]

    # Removes punctuation marks
    # tokens = [token for token in tokens if token not in string.punctuation]

    return tokens

# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = preprocess_text(text)
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

# Process all PDF files in a given directory, returns elapsed time and peak memory
def process_pdfs(data_dir, model, chunk_size=300, overlap=50):

    # Start time / memory check
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

    # returns time and peak memory
    return round(elapsed, 4), round((peak / 1024**2), 2)


def search_embeddings(query, model, top_k=3):

    query_embedding = get_embedding(query, model)

    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        # Perform the search
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        # Transform results into the expected format
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        # Print results for debugging
        # for result in top_results:
        #     print(
        #         f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
        #     )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []
    
def generate_rag_response(query, context_results, model):

    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

    Context:
    {context_str}

    Query: {query}

    Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]



# IMPORT THIS
def run_test(queries, embedding_model, llm_model, chunk_size=300, overlap=50):
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

    clear_redis_store()
    create_hnsw_index()

    print('Processing PDFs...')
    index_elapsed, index_memory = process_pdfs("Slides/", embedding_model, chunk_size, overlap)
    print("\n---Done processing PDFs---\n")

    # define csv file
    csv_filename = "test_results.csv"

    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Write header only if the file has no data
        if file.tell() == 0:
            writer.writerow(["index_elapsed", "index_memory", "query", "query_time_elapsed"])

        for query in queries:
            print('Query:', query)
            start_time = time.time()
            
            # Generate response
            response = generate_rag_response(query, search_embeddings(query, embedding_model), llm_model)
            print(response)

            elapsed = time.time() - start_time
            print(f'Time elapsed: {round(elapsed, 4)} seconds')
            print('---------------------------')

            # Write data row to CSV
            writer.writerow([index_elapsed, index_memory, query, round(elapsed, 4)])

    print(f"Results saved to {csv_filename}")



