#!/usr/bin/env python3
"""
Ad Search API: Offline Indexer (Final Version)
Optimizations:
- Parallel Embedding (8 Workers)
- Hybrid Search (Dense + Sparse)
- POST-INDEXING MERGE (Force 1 Segment) -> Crucial for <10ms latency
"""

import os
import sys
import time
import boto3
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, OptimizersConfigDiff,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    HnswConfigDiff, SparseVectorParams, SparseIndexParams, SparseVector
)
from fastembed import SparseTextEmbedding, TextEmbedding
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
COLLECTION_NAME = "synthetic_ads"
NEW_COLLECTION_NAME = "synthetic_ads_optimized"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
EMBEDDING_DIMENSION = 384
START_BATCH = 1
END_BATCH = 100

# 8 Workers x 4 Threads = 32 Cores Utilized
NUM_WORKERS = 8 

# Environment config
BUCKET_URL = os.environ.get("BUCKET_URL", "https://storage.railway.app")
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
QDRANT_URL = os.environ["QDRANT_URL"]


def create_optimized_collection(qdrant_client: QdrantClient):
    """Create collection with High-Throughput settings."""
    print("🚀 Creating optimized collection...")

    try:
        qdrant_client.delete_collection(NEW_COLLECTION_NAME)
        print(f"✅ Deleted existing collection")
    except:
        pass

    qdrant_client.create_collection(
        collection_name=NEW_COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
                on_disk=False 
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
        on_disk_payload=False,
        optimizers_config=OptimizersConfigDiff(
            # ⚡ INGESTION MODE: Keep segments split for faster upload speed
            default_segment_number=0, 
            memmap_threshold=None,
            max_optimization_threads=8
        ),
        hnsw_config=HnswConfigDiff(
            m=32,
            ef_construct=200,
            full_scan_threshold=10000,
            max_indexing_threads=0,
            on_disk=False
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True
            )
        )
    )
    print(f"✅ Created collection '{NEW_COLLECTION_NAME}'")


def process_single_batch_worker(batch_num: int) -> int:
    """Worker process."""
    # 🛡️ SAFETY: Constrain threads per worker to prevent CPU thrashing
    os.environ["OMP_NUM_THREADS"] = "4"
    
    # Initialize Clients
    s3_client = boto3.client(
        's3',
        endpoint_url=BUCKET_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    qdrant_client = QdrantClient(url=QDRANT_URL, timeout=60.0)
    
    # Load Models
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    # Download
    filename = f"batches/ads_batch_{batch_num:04d}.json"
    try:
        import json
        resp = s3_client.get_object(Bucket=BUCKET_NAME, Key=filename)
        ads = json.loads(resp['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"❌ Batch {batch_num} error: {e}")
        return 0

    if not ads: return 0

    # Extract Texts
    texts = [ad.get("description_long", "").strip() for ad in ads if ad.get("description_long")]
    valid_ads = [ad for ad in ads if ad.get("description_long")]

    if not texts: return 0

    # Embed
    all_dense = []
    # Process in chunks of 256 for CPU cache efficiency
    for i in range(0, len(texts), 256):
        all_dense.extend(list(embedding_model.embed(texts[i:i+256])))
    
    sparse_embeddings = list(sparse_model.embed(texts))

    # Prepare Points
    points = []
    base_id = batch_num * 10000
    for idx, (ad, dense, sparse) in enumerate(zip(valid_ads, all_dense, sparse_embeddings)):
        point = PointStruct(
            id=base_id + idx,
            vector={
                "dense": dense.tolist(),
                "sparse": SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())
            },
            payload={
                "id": ad.get("id"),
                "brand_name": ad.get("brand_name"),
                "title": ad.get("title"),
                "price": ad.get("price"),
                "landing_page": ad.get("landing_page"),
                "bucket_names": ad.get("bucket_names", []),
                "category": ad.get("category"),
                "ad_quality_score": ad.get("ad_quality_score")
            }
        )
        points.append(point)

    # Upload
    for i in range(0, len(points), 100):
        qdrant_client.upsert(
            collection_name=NEW_COLLECTION_NAME,
            points=points[i:i+100],
            wait=False
        )

    return len(points)


def main():
    print("="*80)
    print("🚀 QDRANT OPTIMIZATION (FINAL)")
    print("="*80)

    qdrant_client = QdrantClient(url=QDRANT_URL, timeout=60.0)
    
    # 1. Create Collection
    create_optimized_collection(qdrant_client)

    # 2. Parallel Processing
    print(f"📥 Processing batches {START_BATCH}-{END_BATCH} with {NUM_WORKERS} workers...")
    total_ads = 0
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_batch_worker, b): b for b in range(START_BATCH, END_BATCH + 1)}
        
        for future in as_completed(futures):
            b_num = futures[future]
            try:
                count = future.result()
                total_ads += count
                print(f"   ✅ Batch {b_num} done ({count} ads)")
            except Exception as e:
                print(f"   ❌ Batch {b_num} failed: {e}")

    print(f"\n✅ All batches processed. Total: {total_ads:,}")

    # 3. FORCE MERGE (The Secret Sauce)
    print("\n🧹 Force Merging into 1 Segment (Defrag)...")
    qdrant_client.update_collection(
        collection_name=NEW_COLLECTION_NAME,
        optimizer_config=OptimizersConfigDiff(default_segment_number=1)
    )

    # 4. Wait for Green
    print("⏳ Waiting for optimization...")
    while True:
        info = qdrant_client.get_collection(NEW_COLLECTION_NAME)
        if info.status == "green" and info.segments_count == 1:
            break
        print(f"   Status: {info.status} | Segments: {info.segments_count}")
        time.sleep(2)

    print("\n🎉 DONE! Collection is defragmented and ready for Tier-1 speeds.")

if __name__ == "__main__":
    main()