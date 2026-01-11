#!/usr/bin/env python3
"""
Ad Search API - Streamlit GUI
Interactive interface for hybrid search with latency monitoring
"""

import streamlit as st
import requests
import pandas as pd
from typing import Dict, List, Optional
import time

# API Configuration
API_URL = "https://ad-search-api-production.up.railway.app"

# Available categories
CATEGORIES = [
    "cloud_services",
    "dev_tools",
    "saas_tools",
    "online_education",
    "health_fitness",
    "finance_tools",
    "ecommerce_tools",
    "marketing_tools",
    "productivity_tools",
    "security_tools"
]


def search_ads(query: str, categories: Optional[List[str]] = None, limit: int = 10) -> Dict:
    """Call the search API."""
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/search",
            json={
                "query": query,
                "bucket_names": categories if categories else None,
                "limit": limit
            },
            timeout=10
        )
        client_latency = (time.time() - start_time) * 1000

        response.raise_for_status()
        result = response.json()
        result["client_latency_ms"] = round(client_latency, 2)
        return result
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


def render_latency_metrics(result: Dict):
    """Render latency breakdown with visual metrics."""
    st.subheader("⏱️ Performance Metrics")

    # Main metrics in columns (5 columns now)
    col1, col2, col3, col4, col5 = st.columns(5)

    total_ms = result['latency_ms']
    breakdown = result.get('breakdown', {})

    with col1:
        st.metric(
            "Total Latency",
            f"{total_ms:.1f} ms",
            delta=None
        )

    with col2:
        embedding_ms = breakdown.get('embedding_ms', 0)
        st.metric(
            "Embedding",
            f"{embedding_ms:.1f} ms",
            delta=f"{(embedding_ms/total_ms*100):.0f}%"
        )

    with col3:
        search_ms = breakdown.get('search_fusion_ms', 0)
        st.metric(
            "Search + Fusion",
            f"{search_ms:.1f} ms",
            delta=f"{(search_ms/total_ms*100):.0f}%"
        )

    with col4:
        rerank_ms = breakdown.get('rerank_ms', 0)
        st.metric(
            "Reranking",
            f"{rerank_ms:.1f} ms",
            delta=f"{(rerank_ms/total_ms*100):.0f}%"
        )

    with col5:
        st.metric(
            "Client Total",
            f"{result.get('client_latency_ms', 0):.1f} ms",
            delta="+ network"
        )

    # Detailed breakdown with ALL stages
    if breakdown:
        st.write("**Complete Pipeline Breakdown:**")

        breakdown_data = {
            "Stage": [
                "1. Filter Build",
                "2. Embedding (Dense + Sparse)",
                "3. Search + Fusion (RRF)",
                "4. Reranking (Jina)",
                "5. Format Results"
            ],
            "Time (ms)": [
                breakdown.get('filter_build_ms', 0),
                breakdown.get('embedding_ms', 0),
                breakdown.get('search_fusion_ms', 0),
                breakdown.get('rerank_ms', 0),
                breakdown.get('format_ms', 0)
            ]
        }

        df = pd.DataFrame(breakdown_data)
        st.bar_chart(df.set_index("Stage"))

        # Show percentage breakdown
        st.write("**Time Distribution:**")
        total_accounted = sum(breakdown_data["Time (ms)"])
        percentages = {
            "Filter": f"{(breakdown.get('filter_build_ms', 0)/total_accounted*100):.1f}%",
            "Embedding": f"{(breakdown.get('embedding_ms', 0)/total_accounted*100):.1f}%",
            "Search": f"{(breakdown.get('search_fusion_ms', 0)/total_accounted*100):.1f}%",
            "Rerank": f"{(breakdown.get('rerank_ms', 0)/total_accounted*100):.1f}%",
            "Format": f"{(breakdown.get('format_ms', 0)/total_accounted*100):.1f}%"
        }
        st.json(percentages)


def render_ad_card(ad: Dict, index: int):
    """Render a single ad result as a card."""
    with st.container():
        st.markdown(f"### {index}. {ad.get('brand_name', 'Unknown')} - {ad.get('title', 'No Title')}")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            categories = ad.get('payload', {}).get('bucket_names', [])
            if categories:
                st.write(f"**Categories:** {', '.join(categories)}")

        with col2:
            price = ad.get('price')
            if price is not None:
                st.write(f"**Price:** ${price}")
            else:
                st.write("**Price:** N/A")

        with col3:
            st.write(f"**Relevance:** {ad.get('score', 0):.4f}")
            st.caption("Reranker score")

        # Show landing page if available
        landing_page = ad.get('payload', {}).get('landing_page')
        if landing_page:
            st.write(f"🔗 [Visit Website]({landing_page})")

        st.divider()


def main():
    """Main Streamlit app."""
    # Page config
    st.set_page_config(
        page_title="Ad Search - Hybrid Search",
        page_icon="🔍",
        layout="wide"
    )

    # Title and description
    st.title("🔍 Ad Search API - Hybrid Search Demo")
    st.markdown("""
    **Real-time hybrid search** with Dense (bge-small) + Sparse (BM25) vectors,
    fused using Reciprocal Rank Fusion (RRF), then reranked with FlashRank for maximum precision.
    """)

    # Sidebar for filters
    with st.sidebar:
        st.header("⚙️ Search Settings")

        # Category filter
        selected_categories = st.multiselect(
            "Filter by Categories",
            options=CATEGORIES,
            default=None,
            help="Select one or more categories to filter results"
        )

        # Results limit
        limit = st.slider(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of results to return"
        )

        st.divider()

        # API status
        st.subheader("📊 API Status")
        try:
            health = requests.get(f"{API_URL}/health", timeout=5).json()
            st.success("✅ API Online")
            st.write(f"**Mode:** {health.get('mode', 'N/A')}")
            st.write(f"**Ads:** {health.get('points_count', 0):,}")
            st.write(f"**Connection:** {health.get('connection_type', 'N/A')}")
        except:
            st.error("❌ API Offline")

    # Main search interface
    st.header("🔎 Search Ads")

    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your search query",
            placeholder="e.g., cloud hosting for developers",
            label_visibility="collapsed"
        )

    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)

    # Example queries
    st.write("**Try these:** `cloud hosting`, `online education`, `fitness tracking app`")

    st.divider()

    # Perform search
    if search_button and query:
        with st.spinner("Searching..."):
            result = search_ads(
                query=query,
                categories=selected_categories if selected_categories else None,
                limit=limit
            )

        if result:
            # Show latency metrics
            render_latency_metrics(result)

            st.divider()

            # Show results
            results = result.get('results', [])
            if results:
                st.header(f"📋 Search Results ({len(results)} ads)")

                for i, ad in enumerate(results, 1):
                    render_ad_card(ad, i)
            else:
                st.warning("No results found. Try a different query or remove category filters.")

    elif search_button and not query:
        st.warning("Please enter a search query.")

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
    <small>Powered by bge-small (Dense) + FastEmbed BM25 (Sparse) + FlashRank Reranker | Qdrant Vector Database | FastAPI</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
