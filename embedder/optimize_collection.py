#!/usr/bin/env python3
"""
Optimize Qdrant collection for sub-10ms search latency with HYBRID SEARCH.

This script will:
1. Create a new optimized collection with:
   - HYBRID SEARCH: Dense (bge-small-en-v1.5) + Sparse (BM25) vectors
   - Scalar quantization (4x memory reduction)
   - Optimized HNSW parameters
   - In-memory storage for vectors and index
2. Re-embed all ads using:
   - Dense: BAAI/bge-small-en-v1.5 (local, ~10ms per query)
   - Sparse: Qdrant/bm25 via fastembed (keyword matching)
3. Hybrid search combines semantic + keyword matching

Expected improvements:
- Embedding: ~53ms (Voyage API) → ~10ms (local bge-small)
- Qdrant search: ~45ms → ~35ms (384 vs 512 dimensions)
- Search quality: Maintained (reranker compensates)
- Total: ~190ms → ~137ms
- Indexing: Parallel processing with 8 workers (utilizes 32-core CPU)

Performance:
- With 8 parallel workers: ~15-20 minutes for 100 batches
- Each worker processes batches independently
- Deterministic IDs prevent collisions (batch_num * 10000 + idx)

Usage:
    python optimize_collection.py
"""

import os
import sys
import time
import boto3
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    HnswConfigDiff,
    SparseVectorParams,
    SparseIndexParams,
    SparseVector
)
from fastembed import SparseTextEmbedding, TextEmbedding
from dotenv import load_dotenv

load_dotenv()

# Configuration
COLLECTION_NAME = "synthetic_ads"
NEW_COLLECTION_NAME = "synthetic_ads_optimized"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Fast local dense embedding
SPARSE_MODEL = "Qdrant/bm25"  # BM25 sparse embedding
EMBEDDING_DIMENSION = 384  # bge-small uses 384 dimensions
BATCH_SIZE = 32  # Process 32 ads at a time
START_BATCH = 1
END_BATCH = 100
NUM_WORKERS = 8  # Number of parallel workers (reduce to avoid memory issues)

# AWS S3 Configuration
BUCKET_URL = os.environ.get("BUCKET_URL", "https://storage.railway.app")
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET_NAME = os.environ["BUCKET_NAME"]

# Qdrant Configuration
QDRANT_URL = os.environ["QDRANT_URL"]


def create_optimized_collection(qdrant_client: QdrantClient):
    """Create an optimized Qdrant collection."""
    print("🚀 Creating optimized collection...")
    print()

    # Try to delete existing collection first (if it exists)
    print(f"🗑️  Attempting to delete existing collection '{NEW_COLLECTION_NAME}' if it exists...")
    try:
        qdrant_client.delete_collection(NEW_COLLECTION_NAME)
        print(f"✅ Deleted existing collection")
        time.sleep(2)  # Give Qdrant time to cleanup
    except Exception as e:
        # Collection doesn't exist or other error - that's okay, we'll create it fresh
        print(f"   Collection doesn't exist or already deleted (this is fine)")

    # Create collection with HYBRID SEARCH (dense + sparse)
    qdrant_client.create_collection(
        collection_name=NEW_COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
                on_disk=False  # Keep vectors in RAM for speed
            )
        },
        on_disk_payload=False,
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False  # Keep sparse index in RAM too
                )
            )
        },
        optimizers_config=OptimizersConfigDiff(
            deleted_threshold=0.2,
            vacuum_min_vector_number=1000,
            default_segment_number=1,    # Target 1 segment for optimal search speed
            max_segment_size=200000,     # Allow up to 200K points per segment
            max_optimization_threads=8,  # Limit background work to 8 cores
            indexing_threshold=50000,    # Higher threshold = fewer segments (was 10k)
            flush_interval_sec=5,
            memmap_threshold=None        # Disable memmap (force RAM)
        ),
        hnsw_config=HnswConfigDiff(
            m=32,  # Number of connections per element (optimal for 100K vectors)
            ef_construct=200,  # Construction time vs accuracy tradeoff
            max_indexing_threads=0,
            full_scan_threshold=10000,
            on_disk=False  # Keep HNSW index in RAM for speed
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,  # Use 8-bit integers (4x memory reduction)
                quantile=0.99,  # Use 99th percentile for range
                always_ram=True  # Keep quantized vectors in RAM
            )
        ),
        shard_number=1,
        replication_factor=1
    )

    print(f"✅ Created optimized collection '{NEW_COLLECTION_NAME}'")
    print()
    print("Optimization settings:")
    print("  ✓ HYBRID SEARCH: Dense (bge-small-en-v1.5) + Sparse (BM25)")
    print("  ✓ Vectors in RAM (not disk)")
    print("  ✓ HNSW index in RAM")
    print("  ✓ Scalar quantization (INT8) on dense vectors")
    print("  ✓ M=32, EF_construct=200")
    print("  ✓ 384-dimensional embeddings")
    print("  ✓ Target: 1 segment for optimal search (indexing_threshold=50K)")
    print()


def download_batch_from_s3(s3_client, bucket_name: str, batch_number: int) -> List[Dict]:
    """Download a batch file from S3."""
    filename = f"batches/ads_batch_{batch_number:04d}.json"

    try:
        import json
        response = s3_client.get_object(Bucket=bucket_name, Key=filename)
        content = response['Body'].read().decode('utf-8')
        ads = json.loads(content)
        return ads
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return []


def generate_embeddings(embedding_model: TextEmbedding, texts: List[str]) -> List[List[float]]:
    """Generate embeddings using local FastEmbed model."""
    try:
        # FastEmbed returns a generator of numpy arrays
        embeddings = list(embedding_model.embed(texts))
        # Convert numpy arrays to lists
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        raise e


def process_single_batch_worker(batch_num: int) -> int:
    """
    Worker function that processes a single batch (runs in separate process).
    Each worker initializes its own models and processes one batch.
    """
    import sys
    print(f"[Worker] Starting batch {batch_num}...", flush=True)
    sys.stdout.flush()

    # Initialize models in this worker process
    print(f"[Worker {batch_num}] Loading models...", flush=True)
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    print(f"[Worker {batch_num}] Models loaded", flush=True)

    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        timeout=60.0
    )

    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        endpoint_url=BUCKET_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    # Download batch
    print(f"[Worker {batch_num}] Downloading from S3...", flush=True)
    ads = download_batch_from_s3(s3_client, BUCKET_NAME, batch_num)
    if not ads:
        print(f"[Worker {batch_num}] No ads found", flush=True)
        return 0
    print(f"[Worker {batch_num}] Downloaded {len(ads)} ads", flush=True)

    # Extract texts
    texts = []
    valid_ads = []
    for ad in ads:
        description_long = ad.get("description_long", "").strip()
        if description_long:
            texts.append(description_long)
            valid_ads.append(ad)

    if not texts:
        return 0

    # Generate DENSE embeddings in chunks (better for CPU)
    all_dense_embeddings = []
    CPU_BATCH_SIZE = 256  # Larger batches for parallel workers
    for i in range(0, len(texts), CPU_BATCH_SIZE):
        batch_texts = texts[i:i + CPU_BATCH_SIZE]
        batch_embeddings = generate_embeddings(embedding_model, batch_texts)
        all_dense_embeddings.extend(batch_embeddings)

    # Generate SPARSE embeddings
    sparse_embeddings = list(sparse_model.embed(texts))

    # Prepare points with DETERMINISTIC IDs based on batch number
    points = []
    base_id = batch_num * 10000  # Each batch gets 10,000 ID slots
    for idx, (ad, dense_embedding, sparse_embedding) in enumerate(zip(valid_ads, all_dense_embeddings, sparse_embeddings)):
        sparse_vector = SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist()
        )

        point = PointStruct(
            id=base_id + idx,  # Deterministic ID
            vector={
                "dense": dense_embedding,
                "sparse": sparse_vector
            },
            payload={
                "id": ad.get("id"),
                "brand_name": ad.get("brand_name"),
                "name": ad.get("name"),
                "title": ad.get("title"),
                "description": ad.get("description"),
                "description_long": ad.get("description_long"),
                "price": ad.get("price"),
                "landing_page": ad.get("landing_page"),
                "bucket_names": ad.get("bucket_names", []),
                "category": ad.get("category"),
                "domain": ad.get("domain"),
                "og_image": ad.get("og_image"),
                "ad_quality_score": ad.get("ad_quality_score")
            }
        )
        points.append(point)

    # Upload to Qdrant in chunks
    UPLOAD_CHUNK_SIZE = 100
    total_uploaded = 0
    for chunk_start in range(0, len(points), UPLOAD_CHUNK_SIZE):
        chunk_end = min(chunk_start + UPLOAD_CHUNK_SIZE, len(points))
        chunk_points = points[chunk_start:chunk_end]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                qdrant_client.upsert(
                    collection_name=NEW_COLLECTION_NAME,
                    points=chunk_points,
                    wait=False
                )
                total_uploaded += len(chunk_points)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    print(f"❌ Batch {batch_num} upload failed after {max_retries} attempts: {e}")
                    raise e

    return total_uploaded


def process_and_upload_batch(
    ads: List[Dict],
    embedding_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
    qdrant_client: QdrantClient,
    start_idx: int
) -> int:
    """Process ads batch and upload to Qdrant with HYBRID embeddings (LEGACY - not used in parallel mode)."""
    # Extract description_long for embedding
    texts = []
    valid_ads = []

    for ad in ads:
        description_long = ad.get("description_long", "").strip()
        if description_long:
            texts.append(description_long)
            valid_ads.append(ad)

    if not texts:
        return 0

    # Generate DENSE embeddings in batches (better for CPU)
    # Process in smaller chunks for better CPU performance
    all_dense_embeddings = []
    CPU_BATCH_SIZE = 256  # Larger batches for efficiency
    for i in range(0, len(texts), CPU_BATCH_SIZE):
        batch_texts = texts[i:i + CPU_BATCH_SIZE]
        batch_embeddings = generate_embeddings(embedding_model, batch_texts)
        all_dense_embeddings.extend(batch_embeddings)

    # Generate SPARSE embeddings (BM25) for all texts at once
    sparse_embeddings = list(sparse_model.embed(texts))

    # Prepare points for Qdrant with HYBRID vectors
    points = []
    for idx, (ad, dense_embedding, sparse_embedding) in enumerate(zip(valid_ads, all_dense_embeddings, sparse_embeddings)):
        # Convert sparse embedding to Qdrant SparseVector format
        sparse_vector = SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist()
        )

        point = PointStruct(
            id=start_idx + idx,
            vector={
                "dense": dense_embedding,
                "sparse": sparse_vector
            },
            payload={
                "id": ad.get("id"),
                "brand_name": ad.get("brand_name"),
                "name": ad.get("name"),
                "title": ad.get("title"),
                "description": ad.get("description"),
                "description_long": ad.get("description_long"),
                "price": ad.get("price"),
                "landing_page": ad.get("landing_page"),
                "bucket_names": ad.get("bucket_names", []),
                "category": ad.get("category"),
                "domain": ad.get("domain"),
                "og_image": ad.get("og_image"),
                "ad_quality_score": ad.get("ad_quality_score")
            }
        )
        points.append(point)

    # Upload to Qdrant in smaller chunks with retry logic (reduce memory pressure)
    UPLOAD_CHUNK_SIZE = 100  # Upload 100 points at a time
    total_uploaded = 0

    for chunk_start in range(0, len(points), UPLOAD_CHUNK_SIZE):
        chunk_end = min(chunk_start + UPLOAD_CHUNK_SIZE, len(points))
        chunk_points = points[chunk_start:chunk_end]

        # Retry logic for each chunk
        max_retries = 3
        for attempt in range(max_retries):
            try:
                qdrant_client.upsert(
                    collection_name=NEW_COLLECTION_NAME,
                    points=chunk_points,
                    wait=False  # Don't wait for indexing (faster)
                )
                total_uploaded += len(chunk_points)
                break  # Success, move to next chunk
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️  Upload chunk failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"    Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Upload chunk failed after {max_retries} attempts: {e}")
                    raise e

    return total_uploaded


def main():
    print("="*80)
    print("🚀 QDRANT COLLECTION OPTIMIZATION")
    print("="*80)
    print()
    print(f"Model: {EMBEDDING_MODEL} (faster)")
    print(f"Batches: {START_BATCH} to {END_BATCH}")
    print(f"Target: Sub-10ms Qdrant searches")
    print()

    # Initialize Qdrant client (for collection creation only)
    print("📡 Connecting to Qdrant...")
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        timeout=60.0  # 60 second timeout for Railway network
    )
    print(f"✅ Connected to Qdrant at {QDRANT_URL}")
    print()
    print(f"Note: Each parallel worker will load its own models ({NUM_WORKERS} workers)")
    print()

    # Create optimized collection
    create_optimized_collection(qdrant_client)

    # Process all batches IN PARALLEL
    print(f"📥 Processing {END_BATCH - START_BATCH + 1} batches in parallel with {NUM_WORKERS} workers...")
    print(f"   Using {NUM_WORKERS} CPU cores (~4 cores per worker)")
    print()

    total_ads = 0
    completed_batches = 0
    total_batches = END_BATCH - START_BATCH + 1

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all batches to the pool
        future_to_batch = {
            executor.submit(process_single_batch_worker, batch_num): batch_num
            for batch_num in range(START_BATCH, END_BATCH + 1)
        }

        # Process results as they complete
        for future in as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            try:
                count = future.result()
                total_ads += count
                completed_batches += 1
                progress = (completed_batches / total_batches) * 100
                print(f"✅ Batch {batch_num:3d} complete ({count:4d} ads) | Progress: {completed_batches}/{total_batches} ({progress:.1f}%)")
            except Exception as e:
                print(f"❌ Batch {batch_num} failed: {e}")

    print()
    print(f"✅ All {total_batches} batches processed! Total ads: {total_ads:,}")

    print("⏳ Waiting for indexing to finish...")
    while True:
        info = qdrant_client.get_collection(NEW_COLLECTION_NAME)
        if info.status == "green":
            break
        print(f"   Status: {info.status}... waiting 2s")
        time.sleep(2)
        
    print(f"✅ Optimization Complete! Collection '{NEW_COLLECTION_NAME}' is GREEN and ready.")

    print()
    print("="*80)
    print("✅ OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Total ads embedded: {total_ads:,}")
    print()

    # Get collection stats
    info = qdrant_client.get_collection(NEW_COLLECTION_NAME)
    print(f"📊 Collection Stats:")
    print(f"   Name: {NEW_COLLECTION_NAME}")
    print(f"   Points: {info.points_count:,}")
    print(f"   Status: {info.status}")
    print(f"   Vectors: Dense (bge-small-en-v1.5, 384d) + Sparse (BM25)")
    print()

    # Prompt to switch collections
    print("🔄 Ready to switch collections")
    print()
    print(f"Old collection: {COLLECTION_NAME}")
    print(f"New collection: {NEW_COLLECTION_NAME}")
    print()
    print("Next steps:")
    print("1. Test the new collection first")
    print("2. Update search_api.py to use local bge-small-en-v1.5 model")
    print("3. Redeploy the API")
    print(f"4. Delete old collection: qdrant_client.delete_collection('{COLLECTION_NAME}')")
    print()
    print("🎉 Your search will be ~1.4x faster with no API costs!")


if __name__ == "__main__":
    main()
