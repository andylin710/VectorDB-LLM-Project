import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")


try:
    response = chroma_client.get_or_create_collection(name="my_collection")
    print(response)
except Exception as e:
    print(f"Qdrant connection error: {e}")