# Dockerfile for Qdrant Collection Optimizer (one-time optimization job)
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy optimization script
COPY optimize_collection.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the optimizer
# This creates a new optimized collection with:
# - Scalar quantization (INT8) for 4x memory reduction
# - RAM-only storage for 10-100x faster searches
# - Optimized HNSW parameters (M=16, EF_construct=100)
# - voyage-lite-02-instruct model (3x faster embedding)
# Expected: 327ms → 45ms total latency (7x speedup)
CMD ["python", "optimize_collection.py"]
