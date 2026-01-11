# Ad Search GUI

Interactive web interface for the Ad Search API with real-time latency monitoring.

## Features

- **Hybrid Search**: Dense (Voyage AI) + Sparse (BM25) with RRF fusion
- **Real-time Metrics**: Detailed latency breakdown visualization
- **Category Filtering**: Filter ads by multiple categories
- **Responsive UI**: Clean, modern interface built with Streamlit
- **Performance Monitoring**: Track embedding, search, and total latency

## Quick Start

### Install Dependencies

```bash
cd ui
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## Usage

1. **Enter a search query** in the text box (e.g., "cloud hosting for developers")
2. **Optional**: Select category filters from the sidebar
3. **Adjust** the number of results using the slider
4. **Click Search** to see results with full latency breakdown

## Features Explained

### Performance Metrics
- **Total Latency**: End-to-end API processing time
- **Embedding**: Time to generate dense + sparse vectors (parallel)
- **Search + Fusion**: Qdrant hybrid search with RRF fusion
- **Client Total**: Includes network round-trip time

### Search Results
Each ad shows:
- Brand name and title
- Price (if available)
- RRF score (fusion ranking)
- Categories
- Link to landing page

### API Status
The sidebar displays:
- API health status
- Total number of indexed ads
- Search mode (hybrid-rrf)
- Connection type (HTTP/REST or gRPC)

## Architecture

```
┌─────────────────┐
│  Streamlit UI   │
│   (app.py)      │
└────────┬────────┘
         │ HTTP POST
         │
         ▼
┌─────────────────┐
│   Search API    │
│   (FastAPI)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│ Voyage │ │Qdrant  │
│   AI   │ │ Vector │
│(Dense) │ │  DB    │
└────────┘ └────────┘
    │         │
    └────┬────┘
         │ RRF Fusion
         ▼
    ┌─────────┐
    │ Results │
    └─────────┘
```

## API Endpoint

By default, the app connects to:
```
https://ad-search-api-production.up.railway.app
```

To change the API URL, edit `API_URL` in `app.py`.

## Performance

Typical latency breakdown (as of current deployment):
- **Embedding**: ~70-85ms (Voyage API network call)
- **Search + Fusion**: ~7-10ms (Qdrant hybrid search)
- **Total**: ~90-120ms average

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Production (Railway/Streamlit Cloud)
1. Push to GitHub
2. Connect to Streamlit Cloud or deploy to Railway
3. Set `API_URL` environment variable if needed

## Troubleshooting

**"API Offline" error:**
- Check if the search API is running
- Verify `API_URL` is correct
- Check network connectivity

**Slow responses:**
- Most latency is from Voyage AI embedding generation (~70ms)
- Qdrant search is optimized (<10ms)
- Network latency depends on your location

**No results:**
- Try broader search queries
- Remove category filters
- Check API health status

## Tech Stack

- **Frontend**: Streamlit (Python web framework)
- **API**: FastAPI
- **Vector DB**: Qdrant
- **Dense Embeddings**: Voyage AI (voyage-3-lite)
- **Sparse Embeddings**: FastEmbed BM25
- **Fusion**: Reciprocal Rank Fusion (RRF)
