import hashlib
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Define default arguments for the DAG (retries, alerts, start date)
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    dag_id="vector_ingestion_pipeline",
    default_args=default_args,
    description="Automated pipeline to chunk documents and upsert embeddings into Qdrant",
    schedule="@daily",  # Runs once every day
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ai", "vector", "embedding"],
)
def vector_ingestion_pipeline():

    @task()
    def extract_and_chunk_documents():
        """Task 1: Read raw documents and split them into chunks."""
        # Simulated raw documents (in production, read from S3 or API)
        raw_documents = [
            {
                "file_name": "airflow_guide.txt",
                "text": "Apache Airflow is an open-source platform to programmatically author, schedule, and monitor workflows. It organizes pipelines into Directed Acyclic Graphs."
            },
            {
                "file_name": "qdrant_guide.txt",
                "text": "Qdrant is a vector database written in Rust. It utilizes high-dimensional vectors and HNSW indexing for rapid semantic similarity search and retrieval."
            }
        ]

        def chunk_text(text, chunk_size=150, overlap=30):
            words = text.split()
            chunks = []
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk:
                    chunks.append(chunk)
            return chunks

        processed_chunks = []
        for doc in raw_documents:
            file_name = doc["file_name"]
            chunks = chunk_text(doc["text"])
            for chunk_index, chunk_text_content in enumerate(chunks):
                chunk_hash = hashlib.sha256(chunk_text_content.encode("utf-8")).hexdigest()
                processed_chunks.append({
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    "chunk_hash": chunk_hash,
                    "text": chunk_text_content
                })
        
        print(f"Extracted and chunked {len(processed_chunks)} total segments.")
        return processed_chunks

    @task()
    def embed_and_upsert_to_qdrant(chunks):
        """Task 2: Convert chunks into vectors using local model and store in Qdrant."""
        # Load local HuggingFace embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Connect to Qdrant. 
        # Note: Inside Docker compose, use service name 'qdrant' instead of 'localhost' if running inside a worker container, 
        # but 'http://localhost:6333' works if Airflow runs locally.
        client = QdrantClient(url="http://qdrant:6333")
        collection_name = "production_vector_pipeline"

        # Ensure collection exists
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )

        points_to_upsert = []
        for idx, item in enumerate(chunks, start=1):
            vector = model.encode(item["text"]).tolist()
            points_to_upsert.append(
                models.PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "file_name": item["file_name"],
                        "chunk_index": item["chunk_index"],
                        "chunk_hash": item["chunk_hash"],
                        "text": item["text"]
                    }
                )
            )

        client.upsert(collection_name=collection_name, points=points_to_upsert)
        print(f"Successfully upserted {len(points_to_upsert)} points to Qdrant!")

    # Define the DAG pipeline flow (Task 1 passes data via XCom to Task 2)
    chunks_data = extract_and_chunk_documents()
    embed_and_upsert_to_qdrant(chunks_data)

# Register the DAG
dag_instance = vector_ingestion_pipeline()