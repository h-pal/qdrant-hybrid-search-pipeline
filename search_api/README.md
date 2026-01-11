# Ad Search API Service

FastAPI service providing semantic search over embedded ads using Qdrant vector database.

## Files

### Core Service
- **`search_api.py`** - FastAPI search service
  - Semantic search with Voyage AI embeddings
  - Category filtering by `bucket_names`
  - Latency tracking for performance monitoring
  - Health checks and diagnostics

### Testing
- **`test_search_api.py`** - Comprehensive test suite
  - Single query testing
  - Batch benchmarking
  - Category filter validation
  - Latency statistics

### Configuration
- **`Dockerfile`** - Production deployment
- **`requirements.txt`** - Python dependencies
- **`.env`** - Environment variables

## Environment Variables

Required in `.env`:

```bash
# Qdrant Vector Database
QDRANT_URL=http://qdrant.railway.internal:6333

# Voyage AI Embeddings
VOYAGE_API_KEY=pa-xxx
```

**Note:** Search API only needs these 2 variables (no S3 credentials)

## Deployment to Railway

```bash
# From search_api directory
railway up

# Railway Configuration:
# - Service name: ad-search-api
# - Dockerfile: Dockerfile
# - Restart policy: Always (long-running service)
# - Environment: QDRANT_URL, VOYAGE_API_KEY
# - Networking: Generate public domain
```

**Expected startup:** 5-10 seconds
**Result:** Public API accessible at `https://ad-search-api-production.up.railway.app`

## API Endpoints

### Health Check
```bash
GET /health

# Response:
{
  "status": "healthy",
  "qdrant": "connected",
  "collection": "synthetic_ads",
  "points_count": 99994,
  "voyage_model": "voyage-3"
}
```

### Search (GET)
```bash
GET /search?q=cloud+hosting&bucket_names=cloud_services,dev_tools&limit=5

# Response includes latency breakdown:
{
  "query": "cloud hosting",
  "results": [...],
  "total_results": 5,
  "latency_ms": {
    "embedding": 110.5,
    "filter_build": 0.2,
    "qdrant_search": 206.3,
    "format_results": 0.1,
    "total": 317.1
  }
}
```

### Search (POST)
```bash
POST /search
Content-Type: application/json

{
  "query": "online education",
  "bucket_names": ["online_education", "saas_tools"],
  "limit": 10
}
```

### Categories
```bash
GET /categories

# Returns list of all available categories
```

### Debug Info
```bash
GET /debug/collection-info

# Returns Qdrant collection configuration
# Useful for diagnosing performance issues
```

## Local Testing

```bash
# Set environment variables
export $(cat .env | xargs)

# Run API locally
uvicorn search_api:app --host 0.0.0.0 --port 8000

# In another terminal, run tests
python test_search_api.py
```

## Performance Benchmarking

```bash
# Run comprehensive benchmark
python test_search_api.py

# Output includes:
# - Health check
# - Single query example
# - 10-query benchmark with statistics
# - Category filtering tests
# - Saves results to benchmark_results.json
```

**Expected Results (Optimized):**
```
Server Latency:
  Mean:    45 ms
  Median:  43 ms
  Min:     38 ms
  Max:     52 ms

Embedding:   35 ms  (voyage-lite)
Qdrant:       8 ms  (quantized + RAM)
```

**Current Results (Unoptimized):**
```
Server Latency:
  Mean:   318 ms
  Median: 316 ms

Embedding:  111 ms  (voyage-3)
Qdrant:     207 ms  (no quantization)
```

## Configuration Options

### Switch to Optimized Collection

After running optimizer in `embedder/`:

Edit `search_api.py` lines 34-35:
```python
COLLECTION_NAME = "synthetic_ads_optimized"  # Changed
VOYAGE_MODEL = "voyage-lite-02-instruct"     # Changed
```

Redeploy:
```bash
railway up
```

**Result:** 7x faster searches (327ms → 45ms)

## Available Categories

Filter searches by these categories:
- `online_education`
- `tech_hardware`
- `cloud_services`
- `dev_tools`
- `finance`
- `ecommerce`
- `travel`
- `health_fitness`
- `saas_tools`
- `entertainment`

## Troubleshooting

### 502 Bad Gateway

**Symptoms:**
- All endpoints return 502
- Requests never reach the app

**Solutions:**
1. Check Railway networking is enabled
2. Regenerate public domain in Railway dashboard
3. Verify service is running (check logs)

### Health Check Timeout

**Symptoms:**
- `/health` returns timeout or 503
- Qdrant calls fail

**Solutions:**
1. Verify Qdrant service is running
2. Check `QDRANT_URL` uses internal address
3. Restart Qdrant service if crashed
4. Check Qdrant logs for OOM errors

### Slow Searches (>300ms)

**Symptoms:**
- Latency breakdown shows high Qdrant time
- `qdrant_search` > 200ms

**Solutions:**
1. Run collection optimizer (`embedder/optimize_collection.py`)
2. Enable quantization on existing collection
3. Verify Qdrant has sufficient memory
4. Check if vectors are on disk (should be in RAM)

### Voyage API Errors

**Symptoms:**
- `/search` returns errors
- "Voyage API key invalid" or rate limits

**Solutions:**
1. Verify `VOYAGE_API_KEY` is set correctly
2. Check API key is valid at voyage.ai
3. Monitor rate limits (should auto-retry)

## Interactive Documentation

Once deployed, visit:
```
https://YOUR-URL.railway.app/docs
```

Provides:
- Interactive API testing
- Request/response schemas
- Example queries
- Try-it-out functionality

## Cost Estimate

### Monthly Costs
- **Railway compute:** ~$5/month (always running)
- **Voyage AI queries:** ~$0.06 per 1M tokens
  - ~$0.60 for 10K queries/day
  - ~$6.00 for 100K queries/day
- **Total:** $5-11/month (depends on traffic)

## Integration Examples

### JavaScript
```javascript
const response = await fetch('https://YOUR-URL/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'cloud hosting',
    bucket_names: ['cloud_services'],
    limit: 5
  })
});
const data = await response.json();
console.log(data.results);
```

### Python
```python
import requests

response = requests.post(
    'https://YOUR-URL/search',
    json={
        'query': 'developer tools',
        'bucket_names': ['dev_tools'],
        'limit': 5
    }
)
results = response.json()['results']
```

## Next Steps

After deployment:
1. ✅ Test health endpoint
2. 🔍 Run benchmark: `python test_search_api.py`
3. 📊 Monitor latency in responses
4. 🚀 If slow (>300ms), run optimizer in `embedder/`
5. 🔧 Switch to optimized collection
6. 📈 Re-run benchmark to verify 7x speedup
