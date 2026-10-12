from qdrant_client import QdrantClient
from qdrant_client.http import models

# 1. Initialize the client (connecting to local Docker container on port 6333)
client = QdrantClient(url="http://localhost:6333")

collection_name = "test_vector_collection"

# 2. Recreate collection (Deletes if it exists and creates fresh)
# Note: all-MiniLM-L6-v2 embedding model outputs 384-dimensional vectors
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=384, 
        distance=models.Distance.COSINE
    ),
)
print(f"Successfully created collection: '{collection_name}'")

# 3. Create dummy test embeddings (simulating 2 text chunks converted to 384-dim vectors)
# In production, these numbers will come from SentenceTransformers or an API.
vector_1 = [0.1] * 384
vector_2 = [0.2] * 384

# 4. Insert points (vectors + payload metadata) into Qdrant
client.upsert(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=1,
            vector=vector_1,
            payload={
                "source_file": "doc1.txt", 
                "chunk_index": 0, 
                "text": "Hello world from Data Engineering pipeline!"
            }
        ),
        models.PointStruct(
            id=2,
            vector=vector_2,
            payload={
                "source_file": "doc2.txt", 
                "chunk_index": 0, 
                "text": "Qdrant and Airflow running locally in Docker."
            }
        ),
    ]
)
print("Successfully inserted 2 test vector points!")

# 5. Perform a similarity search query
query_vector = [0.15] * 384
search_results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=1
)

print("\n--- Search Result Test ---")
for scored_point in search_results.points:
    print(f"Found Match! ID: {scored_point.id}, Score: {scored_point.score:.4f}")
    print(f"Payload Text: {scored_point.payload['text']}")