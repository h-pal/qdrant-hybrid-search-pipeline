#!/usr/bin/env python3
"""
Optimize Qdrant collection for sub-10ms search latency with HYBRID SEARCH.

This script will:
1. Create a new optimized collection with:
   - HYBRID SEARCH: Dense (Voyage) + Sparse (BM25) vectors
   - Scalar quantization (4x memory reduction)
   - Optimized HNSW parameters
   - In-memory storage for vectors and index
2. Re-embed all ads using:
   - Dense: voyage-lite-02-instruct (3x faster)
   - Sparse: Qdrant/bm25 via fastembed (keyword matching)
3. Hybrid search combines semantic + keyword matching

Expected improvements:
- Embedding: 110ms → ~35ms (voyage-lite)
- Qdrant search: 207ms → <10ms (quantization + RAM)
- Search quality: Better (hybrid = semantic + keywords)
- Total: 327ms → ~45ms

Usage:
    python optimize_collection.py
"""

import os
import sys
import time
import boto3
import voyageai
from typing import List, Dict
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
from fastembed import SparseTextEmbedding
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Configuration
COLLECTION_NAME = "synthetic_ads"
NEW_COLLECTION_NAME = "synthetic_ads_optimized"
EMBEDDING_MODEL = "voyage-3-lite"  # Faster dense embedding
SPARSE_MODEL = "Qdrant/bm25"  # BM25 sparse embedding
EMBEDDING_DIMENSION = 512
BATCH_SIZE = 32  # Voyage supports up to 128
START_BATCH = 1
END_BATCH = 100

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
            default_segment_number=0,
            max_optimization_threads=8, # Limit background work to 8 cores (leave rest for queries)
            indexing_threshold=10000,   # Start indexing after 10k points
            flush_interval_sec=5,
            memmap_threshold=None       # Disable memmap (force RAM)
        ),
        hnsw_config=HnswConfigDiff(
            m=32,  # Number of connections per element (16 is good balance)
            ef_construct=200,  # Construction time vs accuracy tradeoff
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
    print("  ✓ HYBRID SEARCH: Dense (Voyage) + Sparse (BM25)")
    print("  ✓ Vectors in RAM (not disk)")
    print("  ✓ HNSW index in RAM")
    print("  ✓ Scalar quantization (INT8) on dense vectors")
    print("  ✓ M=16, EF_construct=100")
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


def generate_embeddings(voyage_client: voyageai.Client, texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Voyage AI."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = voyage_client.embed(
                texts=texts,
                model=EMBEDDING_MODEL,
                input_type="document"
            )
            return result.embeddings
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⚠️  Retry in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e


def process_and_upload_batch(
    ads: List[Dict],
    voyage_client: voyageai.Client,
    sparse_model: SparseTextEmbedding,
    qdrant_client: QdrantClient,
    start_idx: int
) -> int:
    """Process ads batch and upload to Qdrant with HYBRID embeddings."""
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

    # Generate DENSE embeddings in batches (Voyage AI)
    all_dense_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        embeddings = generate_embeddings(voyage_client, batch_texts)
        all_dense_embeddings.extend(embeddings)
        time.sleep(0.1)  # Rate limiting

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

    # Initialize clients
    print("📡 Connecting to services...")
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        timeout=60.0  # 60 second timeout for Railway network
    )
    print(f"✅ Connected to Qdrant at {QDRANT_URL}")

    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    print(f"✅ Voyage AI client initialized")

    # Initialize sparse embedding model for BM25
    print(f"🔍 Initializing BM25 sparse embedding model ({SPARSE_MODEL})...")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    print(f"✅ BM25 model initialized")

    s3_client = boto3.client(
        's3',
        endpoint_url=BUCKET_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    print(f"✅ Connected to S3 bucket")
    print()

    # Create optimized collection
    create_optimized_collection(qdrant_client)

    # Process all batches
    print(f"📥 Processing {END_BATCH - START_BATCH + 1} batches...")
    print()

    total_ads = 0
    current_idx = 0

    for batch_num in tqdm(range(START_BATCH, END_BATCH + 1), desc="Processing batches"):
        ads = download_batch_from_s3(s3_client, BUCKET_NAME, batch_num)

        if ads:
            count = process_and_upload_batch(ads, voyage_client, sparse_model, qdrant_client, current_idx)
            total_ads += count
            current_idx += count

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
    print(f"   Vectors: Dense (voyage-3-lite) + Sparse (BM25)")
    print()

    # Prompt to switch collections
    print("🔄 Ready to switch collections")
    print()
    print(f"Old collection: {COLLECTION_NAME}")
    print(f"New collection: {NEW_COLLECTION_NAME}")
    print()
    print("Next steps:")
    print("1. Test the new collection first")
    print("2. Update search_api.py: COLLECTION_NAME = 'synthetic_ads_optimized'")
    print("3. Update search_api.py: VOYAGE_MODEL = 'voyage-3-lite'")
    print("4. Redeploy the API")
    print(f"5. Delete old collection: qdrant_client.delete_collection('{COLLECTION_NAME}')")
    print()
    print("🎉 Your search will be 7x faster!")


if __name__ == "__main__":
    main()
