#!/usr/bin/env python3
"""
Ad Search API: Ultra-Low Latency Hybrid Search with CPU Reranking
Architecture: Parallel Embedding -> RRF Fusion -> CPU Reranking
Pipeline: bge-small-en-v1.5 (Dense) + BM25 (Sparse) -> Qdrant RRF -> FlashRank
"""

import os
import logging
import time
import concurrent.futures
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query as QueryParam
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding, TextEmbedding
from flashrank import Ranker, RerankRequest
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# --- CONFIGURATION ---
COLLECTION_NAME = "synthetic_ads_optimized"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"  # Local embedding model
SPARSE_MODEL = "Qdrant/bm25"

# Optimization: Only fetch fields required for the UI card
# "description_long" is explicitly excluded to save bandwidth
PAYLOAD_FIELDS = [
    "id", "brand_name", "title", "price", 
    "landing_page", "category", "bucket_names", "ad_quality_score", 
]

# --- APP INITIALIZATION ---
app = FastAPI(
    title="Ad Search API (Tier 1 Speed)",
    version="3.0.0",
    description="Hybrid Search with Server-Side Fusion and Parallel Embedding"
)

# Global Clients
qdrant_client: Optional[QdrantClient] = None
dense_model: Optional[TextEmbedding] = None
sparse_model: Optional[SparseTextEmbedding] = None
reranker: Optional[Ranker] = None

# Thread Executor for Parallel Embedding
# NOTE: Keep at 8 until OMP_NUM_THREADS is set in Railway env vars
# After setting OMP_NUM_THREADS=8, can increase to 16 for more concurrent capacity
executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


@app.on_event("startup")
def startup_event():
    """Initialize clients and warm up models on startup."""
    global qdrant_client, dense_model, sparse_model, reranker
    logger.info("🚀 Starting Search API Initialization...")

    # Log threading configuration for verification
    omp_threads = os.environ.get("OMP_NUM_THREADS", "NOT SET")
    mkl_threads = os.environ.get("MKL_NUM_THREADS", "NOT SET")
    logger.info(f"🧵 Thread Config: OMP_NUM_THREADS={omp_threads}, MKL_NUM_THREADS={mkl_threads}")

    # 1. Initialize Qdrant
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    # Check if gRPC should be used (only for external URLs with proper port)
    use_grpc = os.environ.get("QDRANT_USE_GRPC", "false").lower() == "true"

    try:
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=use_grpc,  # Use gRPC only if explicitly enabled
            timeout=30.0  # 30 second timeout for connections
        )
        # Quick connectivity check
        qdrant_client.get_collections()

        connection_type = "gRPC" if use_grpc else "HTTP/REST"
        logger.info(f"✅ Qdrant Client Initialized ({connection_type})")
    except Exception as e:
        logger.error(f"❌ Qdrant Connection Failed: {e}")
        raise e

    # 2. Initialize Dense Embedding Model (Local)
    logger.info(f"Loading Dense Embedding Model ({DENSE_MODEL})...")
    dense_model = TextEmbedding(model_name=DENSE_MODEL)

    # Warmup: Run dummy inference to load model into RAM
    _ = list(dense_model.embed(["warmup query"]))
    logger.info("✅ Dense Model Loaded & Warmed Up (384 dimensions)")

    # 3. Initialize Sparse Model & Warmup
    logger.info("Loading Sparse Model (This may take a few seconds)...")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    # Warmup: Run dummy inference to load model into RAM
    _ = list(sparse_model.query_embed("warmup query"))
    logger.info("✅ Sparse Model Loaded & Warmed Up")

    # 4. Initialize FlashRank Reranker (CPU-based, ultra-fast)
    logger.info("Loading FlashRank Reranker...")
    reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp")
    logger.info("✅ FlashRank Reranker Loaded (ms-marco-TinyBERT-L-2-v2)")


# --- HELPER FUNCTIONS (Run in Threads) ---

def get_dense_embedding(text: str) -> List[float]:
    """Generate Dense Embedding locally using FastEmbed."""
    embeddings = list(dense_model.embed([text]))
    return embeddings[0].tolist()


def get_sparse_embedding(text: str) -> models.SparseVector:
    """Generate Sparse Vector locally on CPU."""
    # FastEmbed returns a generator, fetch the first result
    sparse_raw = list(sparse_model.query_embed(text))[0]
    
    # Convert to Qdrant Native Format
    return models.SparseVector(
        indices=sparse_raw.indices.tolist(),
        values=sparse_raw.values.tolist()
    )


# --- API MODELS ---

class AdResult(BaseModel):
    id: str
    brand_name: Optional[str] = None
    title: str
    price: Optional[Any] = None
    score: float
    payload: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[AdResult]
    latency_ms: float
    breakdown: Optional[Dict[str, float]] = Field(None, description="Optional detailed latency breakdown")

class SearchRequest(BaseModel):
    query: str
    bucket_names: Optional[List[str]] = None
    limit: int = Field(10, le=50)


# --- ENDPOINTS ---

@app.get("/")
def root():
    """Root endpoint - API info."""
    return {
        "service": "Ad Search API",
        "version": "3.0.0",
        "mode": "hybrid-rrf",
        "collection": COLLECTION_NAME,
        "endpoints": {
            "search": "/search",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    """Health check with collection info."""
    try:
        if not qdrant_client:
            return {"status": "initializing"}

        info = qdrant_client.get_collection(COLLECTION_NAME)
        use_grpc = os.environ.get("QDRANT_USE_GRPC", "false").lower() == "true"

        return {
            "status": "healthy",
            "mode": "hybrid-rrf",
            "collection": COLLECTION_NAME,
            "points_count": info.points_count,
            "connection_type": "gRPC" if use_grpc else "HTTP/REST"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


@app.post("/search", response_model=SearchResponse)
def search_ads(request: SearchRequest):
    """
    Execute Hybrid Search with Server-Side Fusion + CPU Reranking.
    1. Parallel: bge-small-en-v1.5 (Dense) + FastEmbed BM25 (Sparse) embeddings
    2. Server: Qdrant RRF Fusion (Dense + Sparse results)
    3. CPU: FlashRank reranking for final precision (10-20x faster than Jina)
    """
    try:
        start_time = time.time()

        # 1. Build Filter (if buckets provided)
        filter_start = time.time()
        search_filter = None
        if request.bucket_names:
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="bucket_names",
                        match=models.MatchAny(any=request.bucket_names)
                    )
                ]
            )
        filter_time = (time.time() - filter_start) * 1000

        # 2. Parallel Embedding Generation
        # This runs Dense (bge-small) and Sparse (BM25) embedding generation in parallel
        embed_start = time.time()
        future_dense = executor.submit(get_dense_embedding, request.query)
        future_sparse = executor.submit(get_sparse_embedding, request.query)

        dense_vector = future_dense.result()
        sparse_vector = future_sparse.result()
        embed_time = (time.time() - embed_start) * 1000

        # 3. Server-Side RRF Fusion (The "Magic" Step)
        # We send both vectors to Qdrant. Qdrant searches both indices,
        # fuses the ranks internally, and returns candidates for reranking.
        # Fetch 3x more results to give reranker better candidates
        search_start = time.time()
        rerank_candidates = 25 #request.limit * 3
        PREFETCH_LIMIT = 50
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    filter=search_filter,
                    limit=PREFETCH_LIMIT, # Prefetch more for better fusion
                    params=models.SearchParams(
                        hnsw_ef=64,  # Balanced accuracy/speed with gRPC
                        exact=False
                    )
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    filter=search_filter,
                    limit=PREFETCH_LIMIT,
                    params=models.SearchParams(
                        hnsw_ef=64,  # Balanced accuracy/speed with gRPC
                        exact=False
                    )
                ),
            ],
            with_payload=PAYLOAD_FIELDS, # Only return fields we need
            limit=rerank_candidates
        )
        search_time = (time.time() - search_start) * 1000

        # 4. CPU Reranking (FlashRank)
        # Rerank RRF results using the original query for final precision
        rerank_start = time.time()

        # Prepare passages list for FlashRank
        passages = []
        points_list = []
        for idx, point in enumerate(search_results.points):
            brand = point.payload.get("brand_name", "")
            title = point.payload.get("title", "")
            doc_text = f"{brand} {title}".strip()
            passages.append({"id": idx, "text": doc_text})
            points_list.append(point)

        # Rerank using FlashRank (10-20x faster than Jina)
        rerank_request = RerankRequest(query=request.query, passages=passages)
        rerank_results = reranker.rerank(rerank_request)

        # Map reranked results back to points (FlashRank returns sorted results with scores)
        scored_points = []
        for result in rerank_results:
            point_idx = int(result['id'])
            score = result['score']
            scored_points.append((score, points_list[point_idx]))

        rerank_time = (time.time() - rerank_start) * 1000

        # 5. Format Top N Results after reranking
        format_start = time.time()
        formatted_results = []
        for score, point in scored_points[:request.limit]:
            formatted_results.append(
                AdResult(
                    id=str(point.id),
                    brand_name=point.payload.get("brand_name"),
                    title=point.payload.get("title"),
                    price=point.payload.get("price"),
                    score=float(score), # Reranker relevance score
                    payload=point.payload
                )
            )
        format_time = (time.time() - format_start) * 1000

        total_latency = (time.time() - start_time) * 1000

        return SearchResponse(
            results=formatted_results,
            latency_ms=round(total_latency, 2),
            breakdown={
                "filter_build_ms": round(filter_time, 2),
                "embedding_ms": round(embed_time, 2),
                "search_fusion_ms": round(search_time, 2),
                "rerank_ms": round(rerank_time, 2),
                "format_ms": round(format_time, 2)
            }
        )

    except Exception as e:
        logger.error(f"Search Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=SearchResponse)
def search_ads_get(
    q: str = QueryParam(..., description="Search query"),
    bucket_names: Optional[str] = QueryParam(None, description="Comma-separated bucket names (e.g., 'dev_tools,cloud_services')"),
    limit: int = QueryParam(10, ge=1, le=50, description="Number of results (1-50)")
):
    """
    GET endpoint for easy browser/curl testing.

    Example:
        GET /search?q=cloud%20hosting&bucket_names=cloud_services,dev_tools&limit=5
    """
    # Parse bucket_names from comma-separated string
    bucket_list = None
    if bucket_names:
        bucket_list = [name.strip() for name in bucket_names.split(",") if name.strip()]

    # Create request and use POST endpoint logic
    request = SearchRequest(
        query=q,
        bucket_names=bucket_list,
        limit=limit
    )

    return search_ads(request)


if __name__ == "__main__":
    import uvicorn
    # Workers=1 is optimal - models are shared, ThreadPoolExecutor handles concurrency
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)