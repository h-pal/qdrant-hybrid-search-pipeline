#!/usr/bin/env python3
"""Check Qdrant collection configuration and indexing status."""

import os
from qdrant_client import QdrantClient
import json

# Connect to Qdrant
client = QdrantClient(url=os.environ['QDRANT_URL'])

# Get collection info
collection_info = client.get_collection('synthetic_ads')

print("="*80)
print("QDRANT COLLECTION CONFIGURATION")
print("="*80)
print()

print(f"Status: {collection_info.status}")
print(f"Points Count: {collection_info.points_count:,}")
print(f"Vectors Count: {collection_info.vectors_count:,}")
print(f"Indexed Vectors: {collection_info.indexed_vectors_count:,}")
print()

# Check if fully indexed
if collection_info.vectors_count == collection_info.indexed_vectors_count:
    print("✅ All vectors are indexed (HNSW ready)")
else:
    print(f"⚠️  WARNING: {collection_info.vectors_count - collection_info.indexed_vectors_count:,} vectors not indexed yet!")
print()

# Vector config
vectors_config = collection_info.config.params.vectors
print("Vector Configuration:")
print(f"  Dimension: {vectors_config.size}")
print(f"  Distance: {vectors_config.distance}")
print(f"  On Disk: {getattr(vectors_config, 'on_disk', False)}")
print()

# HNSW config
if collection_info.config.hnsw_config:
    hnsw = collection_info.config.hnsw_config
    print("HNSW Configuration:")
    print(f"  M (connections): {hnsw.m}")
    print(f"  EF Construct: {hnsw.ef_construct}")
    print(f"  On Disk: {getattr(hnsw, 'on_disk', False)}")
    print()
else:
    print("⚠️  WARNING: No HNSW configuration found!")
    print()

# Quantization config
if collection_info.config.quantization_config:
    print("Quantization:")
    print(f"  Enabled: Yes")
    print(f"  Config: {collection_info.config.quantization_config}")
else:
    print("Quantization:")
    print(f"  Enabled: No (⚠️  Not optimized!)")
print()

# Optimizer status
if collection_info.config.optimizer_config:
    opt = collection_info.config.optimizer_config
    print("Optimizer Config:")
    print(f"  Indexing Threshold: {opt.indexing_threshold}")
    print(f"  Max Optimization Threads: {opt.max_optimization_threads}")
print()

print("="*80)
print("RECOMMENDATIONS")
print("="*80)
print()

# Check for issues
issues = []
recommendations = []

if collection_info.vectors_count != collection_info.indexed_vectors_count:
    issues.append("Not all vectors are indexed")
    recommendations.append("Wait for indexing to complete or check optimizer status")

if not collection_info.config.quantization_config:
    issues.append("No quantization configured (vectors stored at full precision)")
    recommendations.append("Enable scalar quantization to reduce memory and improve speed")

if getattr(vectors_config, 'on_disk', False):
    issues.append("Vectors stored on disk (slower than RAM)")
    recommendations.append("Keep vectors in RAM for better performance")

if collection_info.config.hnsw_config and getattr(collection_info.config.hnsw_config, 'on_disk', False):
    issues.append("HNSW index stored on disk (slower than RAM)")
    recommendations.append("Keep HNSW index in RAM for sub-10ms searches")

if issues:
    print("❌ Issues Found:")
    for issue in issues:
        print(f"   - {issue}")
    print()
    print("💡 Recommendations:")
    for rec in recommendations:
        print(f"   - {rec}")
else:
    print("✅ Configuration looks good!")

print()
print("="*80)
