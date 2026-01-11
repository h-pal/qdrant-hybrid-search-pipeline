#!/usr/bin/env python3
"""
Test script for Ad Search API - HYBRID SEARCH + CPU RERANKING

Pipeline: Dense (Voyage) + Sparse (BM25) → RRF Fusion → Jina Reranker

Features:
- Single query testing with hybrid search + reranking
- Batch query benchmarking
- Latency statistics (embedding, search+fusion, rerank breakdown)
- Category filtering tests
- Result validation with reranker scores

Usage:
    python test_search_api.py
"""

import requests
import time
import json
from typing import List, Dict, Optional
from statistics import mean, median, stdev
from datetime import datetime

# API Configuration
API_URL = "https://ad-search-api-production.up.railway.app"

# Test queries
TEST_QUERIES = [
    "cloud hosting for developers",
    "online education platform",
    "project management software",
    "fitness tracking app",
    "e-commerce store builder",
    "video streaming service",
    "data analytics tool",
    "mobile payment solution",
    "cybersecurity software",
    "AI-powered chatbot"
]


class SearchAPITester:
    """Test and benchmark the Ad Search API."""

    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.session = requests.Session()  # Reuse connection

    def health_check(self) -> Dict:
        """Check API health status."""
        print("🏥 Performing health check...")
        response = self.session.get(f"{self.api_url}/health")
        response.raise_for_status()
        health = response.json()

        print(f"✅ Status: {health['status']}")
        print(f"📊 Collection: {health['collection']} ({health['points_count']:,} ads)")
        print(f"🎯 Mode: {health['mode']}")
        print(f"🔌 Connection: {health['connection_type']}")
        print()
        return health

    def search(
        self,
        query: str,
        bucket_names: Optional[List[str]] = None,
        limit: int = 5
    ) -> Dict:
        """
        Perform a single search query.

        Args:
            query: Search query string
            bucket_names: Optional list of categories to filter
            limit: Number of results to return

        Returns:
            Search response with results and latency
        """
        # Measure client-side latency
        start_time = time.time()

        response = self.session.post(
            f"{self.api_url}/search",
            json={
                "query": query,
                "bucket_names": bucket_names,
                "limit": limit
            }
        )

        end_time = time.time()
        client_latency = (end_time - start_time) * 1000  # Convert to ms

        response.raise_for_status()
        result = response.json()
        result["client_latency_ms"] = round(client_latency, 2)

        return result

    def print_search_results(self, result: Dict):
        """Pretty print search results."""
        print(f"🔍 Query: \"{result.get('query', 'N/A')}\"")
        print(f"📊 Results: {len(result['results'])}")
        print()

        # Print latency breakdown (hybrid search + reranking structure)
        print(f"⏱️  Total Latency: {result['latency_ms']:.2f} ms")
        if result.get('breakdown'):
            breakdown = result['breakdown']
            print("   Breakdown:")
            print(f"      Filter Build:    {breakdown.get('filter_build_ms', 0):>7.2f} ms")
            print(f"      Embedding:       {breakdown.get('embedding_ms', 0):>7.2f} ms")
            print(f"      Search + Fusion: {breakdown.get('search_fusion_ms', 0):>7.2f} ms")
            print(f"      Rerank (Jina):   {breakdown.get('rerank_ms', 0):>7.2f} ms")
            print(f"      Format Results:  {breakdown.get('format_ms', 0):>7.2f} ms")
        print(f"   Client Total:    {result['client_latency_ms']:>7.2f} ms (includes network)")
        print()

        # Print top results
        print("🎯 Top Results:")
        for i, ad in enumerate(result['results'][:3], 1):
            print(f"\n{i}. {ad['brand_name']} - {ad['title']}")
            print(f"   Price: ${ad.get('price', 'N/A')}")
            print(f"   Rerank Score: {ad['score']:.4f}")
            # Get categories from payload
            categories = ad.get('payload', {}).get('bucket_names', [])
            if categories:
                print(f"   Categories: {', '.join(categories)}")
        print("\n" + "="*80 + "\n")

    def benchmark_queries(
        self,
        queries: List[str],
        bucket_names: Optional[List[str]] = None,
        limit: int = 5
    ) -> Dict:
        """
        Run multiple queries and collect latency statistics.

        Args:
            queries: List of search queries
            bucket_names: Optional category filter
            limit: Number of results per query

        Returns:
            Statistics dictionary
        """
        print(f"🚀 Running benchmark with {len(queries)} queries...")
        print(f"   Limit: {limit} results per query")
        if bucket_names:
            print(f"   Filters: {', '.join(bucket_names)}")
        print()

        results = []
        server_latencies = []
        client_latencies = []
        embedding_latencies = []
        search_fusion_latencies = []
        rerank_latencies = []

        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] Testing: \"{query}\"")

            try:
                result = self.search(query, bucket_names, limit)
                results.append(result)

                server_latencies.append(result['latency_ms'])
                client_latencies.append(result['client_latency_ms'])

                # Extract breakdown metrics if available
                if result.get('breakdown'):
                    embedding_latencies.append(result['breakdown'].get('embedding_ms', 0))
                    search_fusion_latencies.append(result['breakdown'].get('search_fusion_ms', 0))
                    rerank_latencies.append(result['breakdown'].get('rerank_ms', 0))

                print(f"   Server: {result['latency_ms']:.2f}ms | "
                      f"Client: {result['client_latency_ms']:.2f}ms | "
                      f"Results: {len(result['results'])}")

            except Exception as e:
                print(f"   ❌ Error: {str(e)}")

        print("\n" + "="*80 + "\n")

        # Calculate statistics
        stats = {
            "total_queries": len(queries),
            "successful_queries": len(results),
            "timestamp": datetime.now().isoformat(),
            "server_latency_ms": {
                "mean": round(mean(server_latencies), 2) if server_latencies else 0,
                "median": round(median(server_latencies), 2) if server_latencies else 0,
                "min": round(min(server_latencies), 2) if server_latencies else 0,
                "max": round(max(server_latencies), 2) if server_latencies else 0,
                "stdev": round(stdev(server_latencies), 2) if len(server_latencies) > 1 else 0
            },
            "client_latency_ms": {
                "mean": round(mean(client_latencies), 2) if client_latencies else 0,
                "median": round(median(client_latencies), 2) if client_latencies else 0,
                "min": round(min(client_latencies), 2) if client_latencies else 0,
                "max": round(max(client_latencies), 2) if client_latencies else 0,
                "stdev": round(stdev(client_latencies), 2) if len(client_latencies) > 1 else 0
            },
            "embedding_latency_ms": {
                "mean": round(mean(embedding_latencies), 2) if embedding_latencies else 0,
                "median": round(median(embedding_latencies), 2) if embedding_latencies else 0
            },
            "search_fusion_latency_ms": {
                "mean": round(mean(search_fusion_latencies), 2) if search_fusion_latencies else 0,
                "median": round(median(search_fusion_latencies), 2) if search_fusion_latencies else 0
            },
            "rerank_latency_ms": {
                "mean": round(mean(rerank_latencies), 2) if rerank_latencies else 0,
                "median": round(median(rerank_latencies), 2) if rerank_latencies else 0
            }
        }

        return stats

    def print_benchmark_stats(self, stats: Dict):
        """Pretty print benchmark statistics."""
        print("📈 BENCHMARK STATISTICS")
        print("="*80)
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Successful: {stats['successful_queries']}")
        print(f"Timestamp: {stats['timestamp']}")
        print()

        print("⏱️  SERVER LATENCY (Processing Time):")
        server = stats['server_latency_ms']
        print(f"   Mean:   {server['mean']:>7.2f} ms")
        print(f"   Median: {server['median']:>7.2f} ms")
        print(f"   Min:    {server['min']:>7.2f} ms")
        print(f"   Max:    {server['max']:>7.2f} ms")
        print(f"   StdDev: {server['stdev']:>7.2f} ms")
        print()

        print("🌐 CLIENT LATENCY (End-to-End):")
        client = stats['client_latency_ms']
        print(f"   Mean:   {client['mean']:>7.2f} ms")
        print(f"   Median: {client['median']:>7.2f} ms")
        print(f"   Min:    {client['min']:>7.2f} ms")
        print(f"   Max:    {client['max']:>7.2f} ms")
        print(f"   StdDev: {client['stdev']:>7.2f} ms")
        print()

        print("🤖 EMBEDDING LATENCY (Dense + Sparse):")
        emb = stats['embedding_latency_ms']
        print(f"   Mean:   {emb['mean']:>7.2f} ms")
        print(f"   Median: {emb['median']:>7.2f} ms")
        print()

        print("🔍 SEARCH + FUSION LATENCY (Qdrant RRF):")
        search = stats['search_fusion_latency_ms']
        print(f"   Mean:   {search['mean']:>7.2f} ms")
        print(f"   Median: {search['median']:>7.2f} ms")
        print()

        print("🎯 RERANK LATENCY (Jina TextCrossEncoder):")
        rerank = stats['rerank_latency_ms']
        print(f"   Mean:   {rerank['mean']:>7.2f} ms")
        print(f"   Median: {rerank['median']:>7.2f} ms")
        print()

        # Calculate percentages
        total_mean = server['mean']
        emb_pct = (emb['mean'] / total_mean * 100) if total_mean > 0 else 0
        search_pct = (search['mean'] / total_mean * 100) if total_mean > 0 else 0
        rerank_pct = (rerank['mean'] / total_mean * 100) if total_mean > 0 else 0
        other_pct = 100 - emb_pct - search_pct - rerank_pct

        print("📊 LATENCY BREAKDOWN:")
        print(f"   Embedding:        {emb_pct:>5.1f}%")
        print(f"   Search + Fusion:  {search_pct:>5.1f}%")
        print(f"   Rerank:           {rerank_pct:>5.1f}%")
        print(f"   Other:            {other_pct:>5.1f}%")
        print()
        print("="*80)

    def test_categories(self):
        """Test search with different category filters."""
        print("\n🏷️  TESTING CATEGORY FILTERING")
        print("="*80 + "\n")

        tests = [
            ("cloud hosting", ["cloud_services"]),
            ("developer tools", ["dev_tools", "saas_tools"]),
            ("online learning", ["online_education"]),
            ("fitness tracker", ["health_fitness"])
        ]

        for query, categories in tests:
            result = self.search(query, bucket_names=categories, limit=3)
            print(f"Query: \"{query}\" | Categories: {categories}")
            print(f"Results: {len(result['results'])} | Latency: {result['latency_ms']:.2f}ms")

            # Verify all results have at least one matching category
            all_match = all(
                any(cat in ad.get('payload', {}).get('bucket_names', []) for cat in categories)
                for ad in result['results']
            )
            print(f"✅ Filter validation: {'PASS' if all_match else 'FAIL'}")
            print()


def main():
    """Run comprehensive API tests for Hybrid Search + Reranking API."""
    tester = SearchAPITester()

    print("="*80)
    print("🚀 AD SEARCH API - HYBRID SEARCH + RERANKING BENCHMARKING")
    print("="*80)
    print()

    # 1. Health check
    try:
        tester.health_check()
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return

    # 2. Single query example
    print("📝 SINGLE QUERY TEST")
    print("="*80 + "\n")
    try:
        result = tester.search(
            query="cloud hosting for developers",
            bucket_names=["cloud_services", "dev_tools"],
            limit=5
        )
        tester.print_search_results(result)
    except Exception as e:
        print(f"❌ Search failed: {e}\n")

    # 3. Benchmark with multiple queries
    print("🏃 BENCHMARK TEST")
    print("="*80 + "\n")
    try:
        stats = tester.benchmark_queries(TEST_QUERIES, limit=5)
        tester.print_benchmark_stats(stats)

        # Save stats to file
        with open('benchmark_results.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print("\n💾 Results saved to: benchmark_results.json\n")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}\n")

    # 4. Test category filtering
    try:
        tester.test_categories()
    except Exception as e:
        print(f"❌ Category test failed: {e}\n")

    print("✅ All tests completed!")


if __name__ == "__main__":
    main()
