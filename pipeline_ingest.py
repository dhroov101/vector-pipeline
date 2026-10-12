import hashlib
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# 1. Initialize the HuggingFace Embedding Model (Outputs 384-dimensional vectors)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Initialize Qdrant Client connection
client = QdrantClient(url="http://localhost:6333")
collection_name = "production_vector_pipeline"

# 3. Ensure the collection exists with the correct vector size (384)
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=384, 
        distance=models.Distance.COSINE
    ),
)
print(f"Collection '{collection_name}' ready.")

# 4. Simulate reading raw documents (In production, this would read from a folder or S3 bucket)
raw_documents = [
    {
        "file_name": "data_engineering_guide.txt",
        "text": "Data engineering is the practice of designing and building systems for collecting, storing, and analyzing data at scale. It requires mastery over pipelines, SQL, and orchestration tools like Apache Airflow."
    },
    {
        "file_name": "vector_databases_101.txt",
        "text": "Vector databases store high-dimensional numerical embeddings to enable semantic search and Retrieval-Augmented Generation workflows. Qdrant provides high performance through Rust and HNSW indexing."
    }
]

# 5. Deterministic Text Chunking & Hashing Function
def chunk_text(text, chunk_size=150, overlap=30):
    """Splits raw text into sliding-window chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# 6. Process, Embed, and Upsert Points into Qdrant
points_to_upsert = []
point_id = 1

for doc in raw_documents:
    file_name = doc["file_name"]
    chunks = chunk_text(doc["text"])
    
    for chunk_index, chunk_text_content in enumerate(chunks):
        # Generate SHA-256 hash for idempotency & deduplication
        chunk_hash = hashlib.sha256(chunk_text_content.encode("utf-8")).hexdigest()
        
        # Generate real vector embedding using HuggingFace model
        vector = model.encode(chunk_text_content).tolist()
        
        # Build the Qdrant PointStruct containing vector + payload metadata
        points_to_upsert.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    "chunk_hash": chunk_hash,
                    "text": chunk_text_content
                }
            )
        )
        point_id += 1

# Batch upsert points into Qdrant database
client.upsert(
    collection_name=collection_name,
    points=points_to_upsert
)

print(f"Successfully processed and embedded {len(points_to_upsert)} chunks into Qdrant!")

# 7. Test Semantic Search
query = "How does Apache Airflow help with data pipelines?"
query_vector = model.encode(query).tolist()

search_results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=1
)

print("\n--- Semantic Search Test ---")
print(f"Query: '{query}'")
for scored_point in search_results.points:
    print(f"Top Match Score: {scored_point.score:.4f}")
    print(f"Retrieved Text: {scored_point.payload['text']}")