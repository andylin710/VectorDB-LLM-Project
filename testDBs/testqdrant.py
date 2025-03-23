from qdrant_client import QdrantClient

qdrant_client = QdrantClient(url="http://localhost:6333")

try:
    response = qdrant_client.get_collections()
    print(response)
except Exception as e:
    print(f"Qdrant connection error: {e}")
