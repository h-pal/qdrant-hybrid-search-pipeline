#!/usr/bin/env python3
"""
Test script for Ad Search API - HYBRID SEARCH + CPU RERANKING

Pipeline: bge-small (Dense) + BM25 (Sparse) → RRF Fusion → FlashRank

Features:
- Single query testing with hybrid search + reranking
- Batch query benchmarking (10 or 1000 queries)
- Percentile latency statistics (p50, p90, p95, p99)
- Latency breakdown (embedding, search+fusion, rerank)
- Category filtering tests
- Result validation with reranker scores

Usage:
    python test_search_api.py              # Run quick test (10 queries)
    python test_search_api.py --full       # Run full benchmark (1000 queries)
"""

import requests
import time
import json
import sys
from typing import List, Dict, Optional
from statistics import mean, median, stdev
from datetime import datetime
import random

# API Configuration
API_URL = "https://ad-search-api-production.up.railway.app"

# Small test queries for quick tests
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

# Generate 1000 diverse queries for comprehensive benchmarking
def generate_1000_queries() -> List[str]:
    """Generate 1000 realistic ad search queries."""

    # Query templates and variations
    products = [
        "cloud hosting", "project management", "CRM software", "video editing",
        "email marketing", "social media", "e-commerce platform", "website builder",
        "accounting software", "HR management", "inventory system", "payment gateway",
        "analytics tool", "chatbot", "VPN service", "antivirus", "backup solution",
        "password manager", "form builder", "scheduling software", "CMS platform",
        "learning management", "video conferencing", "helpdesk software", "survey tool",
        "SEO tool", "graphic design", "photo editing", "video streaming",
        "podcast hosting", "music streaming", "fitness app", "meal planning",
        "meditation app", "sleep tracker", "workout planner", "nutrition guide",
        "booking system", "appointment scheduler", "ticketing platform", "event management",
        "invoicing software", "time tracking", "expense management", "payroll system",
        "recruiting platform", "applicant tracking", "onboarding software", "performance review",
        "collaboration tool", "knowledge base", "wiki platform", "documentation tool",
        "API management", "monitoring service", "logging platform", "error tracking",
        "database hosting", "file storage", "CDN service", "container platform"
    ]

    descriptors = [
        "affordable", "enterprise", "small business", "startup", "professional",
        "easy to use", "powerful", "secure", "fast", "reliable",
        "cloud-based", "open source", "free", "premium", "scalable",
        "mobile-friendly", "collaborative", "automated", "integrated", "customizable",
        "AI-powered", "real-time", "advanced", "simple", "intuitive",
        "modern", "comprehensive", "flexible", "robust", "efficient"
    ]

    use_cases = [
        "for developers", "for teams", "for enterprises", "for startups", "for freelancers",
        "for small business", "for marketing", "for sales", "for support", "for HR",
        "for finance", "for healthcare", "for education", "for ecommerce", "for SaaS",
        "for agencies", "for consultants", "for remote teams", "for productivity", "for collaboration",
        "for content creators", "for bloggers", "for podcasters", "for YouTubers", "for influencers",
        "for fitness enthusiasts", "for athletes", "for gym owners", "for trainers", "for coaches"
    ]

    queries = []

    # Generate queries with different patterns
    for i in range(1000):
        pattern = i % 10

        if pattern == 0:
            # Simple product query
            query = random.choice(products)
        elif pattern == 1:
            # Product + descriptor
            query = f"{random.choice(descriptors)} {random.choice(products)}"
        elif pattern == 2:
            # Product + use case
            query = f"{random.choice(products)} {random.choice(use_cases)}"
        elif pattern == 3:
            # Descriptor + product + use case
            query = f"{random.choice(descriptors)} {random.choice(products)} {random.choice(use_cases)}"
        elif pattern == 4:
            # Question format
            query = f"best {random.choice(products)}"
        elif pattern == 5:
            # Comparison format
            query = f"{random.choice(products)} vs {random.choice(products)}"
        elif pattern == 6:
            # Feature-focused
            query = f"{random.choice(products)} with {random.choice(descriptors)} features"
        elif pattern == 7:
            # Price-focused
            query = f"cheap {random.choice(products)}"
        elif pattern == 8:
            # Problem-solution
            query = f"how to {random.choice(products)}"
        else:
            # Alternative format
            query = f"top {random.choice(products)} {random.choice(use_cases)}"

        queries.append(query)

    return queries

# Generate the 1000 queries
BENCHMARK_1000_QUERIES = generate_1000_queries()


def percentile(data: List[float], p: float) -> float:
    """
    Calculate the p-th percentile of data.

    Args:
        data: List of numerical values
        p: Percentile (0-100)

    Returns:
        The value at the p-th percentile
    """
    if not data:
        return 0.0

    sorted_data = sorted(data)
    n = len(sorted_data)
    k = (n - 1) * (p / 100)
    f = int(k)
    c = k - f

    if f + 1 < n:
        return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
    else:
        return sorted_data[f]


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

        # Calculate statistics with percentiles
        stats = {
            "total_queries": len(queries),
            "successful_queries": len(results),
            "timestamp": datetime.now().isoformat(),
            "server_latency_ms": {
                "mean": round(mean(server_latencies), 2) if server_latencies else 0,
                "median": round(median(server_latencies), 2) if server_latencies else 0,
                "p50": round(percentile(server_latencies, 50), 2) if server_latencies else 0,
                "p90": round(percentile(server_latencies, 90), 2) if server_latencies else 0,
                "p95": round(percentile(server_latencies, 95), 2) if server_latencies else 0,
                "p99": round(percentile(server_latencies, 99), 2) if server_latencies else 0,
                "min": round(min(server_latencies), 2) if server_latencies else 0,
                "max": round(max(server_latencies), 2) if server_latencies else 0,
                "stdev": round(stdev(server_latencies), 2) if len(server_latencies) > 1 else 0
            },
            "client_latency_ms": {
                "mean": round(mean(client_latencies), 2) if client_latencies else 0,
                "median": round(median(client_latencies), 2) if client_latencies else 0,
                "p50": round(percentile(client_latencies, 50), 2) if client_latencies else 0,
                "p90": round(percentile(client_latencies, 90), 2) if client_latencies else 0,
                "p95": round(percentile(client_latencies, 95), 2) if client_latencies else 0,
                "p99": round(percentile(client_latencies, 99), 2) if client_latencies else 0,
                "min": round(min(client_latencies), 2) if client_latencies else 0,
                "max": round(max(client_latencies), 2) if client_latencies else 0,
                "stdev": round(stdev(client_latencies), 2) if len(client_latencies) > 1 else 0
            },
            "embedding_latency_ms": {
                "mean": round(mean(embedding_latencies), 2) if embedding_latencies else 0,
                "median": round(median(embedding_latencies), 2) if embedding_latencies else 0,
                "p50": round(percentile(embedding_latencies, 50), 2) if embedding_latencies else 0,
                "p90": round(percentile(embedding_latencies, 90), 2) if embedding_latencies else 0,
                "p95": round(percentile(embedding_latencies, 95), 2) if embedding_latencies else 0,
                "p99": round(percentile(embedding_latencies, 99), 2) if embedding_latencies else 0
            },
            "search_fusion_latency_ms": {
                "mean": round(mean(search_fusion_latencies), 2) if search_fusion_latencies else 0,
                "median": round(median(search_fusion_latencies), 2) if search_fusion_latencies else 0,
                "p50": round(percentile(search_fusion_latencies, 50), 2) if search_fusion_latencies else 0,
                "p90": round(percentile(search_fusion_latencies, 90), 2) if search_fusion_latencies else 0,
                "p95": round(percentile(search_fusion_latencies, 95), 2) if search_fusion_latencies else 0,
                "p99": round(percentile(search_fusion_latencies, 99), 2) if search_fusion_latencies else 0
            },
            "rerank_latency_ms": {
                "mean": round(mean(rerank_latencies), 2) if rerank_latencies else 0,
                "median": round(median(rerank_latencies), 2) if rerank_latencies else 0,
                "p50": round(percentile(rerank_latencies, 50), 2) if rerank_latencies else 0,
                "p90": round(percentile(rerank_latencies, 90), 2) if rerank_latencies else 0,
                "p95": round(percentile(rerank_latencies, 95), 2) if rerank_latencies else 0,
                "p99": round(percentile(rerank_latencies, 99), 2) if rerank_latencies else 0
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
        print(f"   Mean:    {server['mean']:>7.2f} ms")
        print(f"   Median:  {server['median']:>7.2f} ms")
        print(f"   P50:     {server['p50']:>7.2f} ms")
        print(f"   P90:     {server['p90']:>7.2f} ms")
        print(f"   P95:     {server['p95']:>7.2f} ms")
        print(f"   P99:     {server['p99']:>7.2f} ms")
        print(f"   Min:     {server['min']:>7.2f} ms")
        print(f"   Max:     {server['max']:>7.2f} ms")
        print(f"   StdDev:  {server['stdev']:>7.2f} ms")
        print()

        print("🌐 CLIENT LATENCY (End-to-End):")
        client = stats['client_latency_ms']
        print(f"   Mean:    {client['mean']:>7.2f} ms")
        print(f"   Median:  {client['median']:>7.2f} ms")
        print(f"   P50:     {client['p50']:>7.2f} ms")
        print(f"   P90:     {client['p90']:>7.2f} ms")
        print(f"   P95:     {client['p95']:>7.2f} ms")
        print(f"   P99:     {client['p99']:>7.2f} ms")
        print(f"   Min:     {client['min']:>7.2f} ms")
        print(f"   Max:     {client['max']:>7.2f} ms")
        print(f"   StdDev:  {client['stdev']:>7.2f} ms")
        print()

        print("🤖 EMBEDDING LATENCY (Dense + Sparse):")
        emb = stats['embedding_latency_ms']
        print(f"   Mean:    {emb['mean']:>7.2f} ms")
        print(f"   Median:  {emb['median']:>7.2f} ms")
        print(f"   P90:     {emb['p90']:>7.2f} ms")
        print(f"   P95:     {emb['p95']:>7.2f} ms")
        print(f"   P99:     {emb['p99']:>7.2f} ms")
        print()

        print("🔍 SEARCH + FUSION LATENCY (Qdrant RRF):")
        search = stats['search_fusion_latency_ms']
        print(f"   Mean:    {search['mean']:>7.2f} ms")
        print(f"   Median:  {search['median']:>7.2f} ms")
        print(f"   P90:     {search['p90']:>7.2f} ms")
        print(f"   P95:     {search['p95']:>7.2f} ms")
        print(f"   P99:     {search['p99']:>7.2f} ms")
        print()

        print("🎯 RERANK LATENCY (FlashRank):")
        rerank = stats['rerank_latency_ms']
        print(f"   Mean:    {rerank['mean']:>7.2f} ms")
        print(f"   Median:  {rerank['median']:>7.2f} ms")
        print(f"   P90:     {rerank['p90']:>7.2f} ms")
        print(f"   P95:     {rerank['p95']:>7.2f} ms")
        print(f"   P99:     {rerank['p99']:>7.2f} ms")
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
    # Check for --full flag
    run_full_benchmark = "--full" in sys.argv or "-f" in sys.argv

    tester = SearchAPITester()

    print("="*80)
    print("🚀 AD SEARCH API - HYBRID SEARCH + RERANKING BENCHMARKING")
    if run_full_benchmark:
        print("   MODE: FULL BENCHMARK (1000 queries)")
    else:
        print("   MODE: QUICK TEST (10 queries)")
    print("="*80)
    print()

    # 1. Health check
    try:
        tester.health_check()
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return

    # 2. Single query example
    if not run_full_benchmark:
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
        # Choose query set based on mode
        if run_full_benchmark:
            queries = BENCHMARK_1000_QUERIES
            print(f"⚡ Running FULL benchmark with {len(queries)} queries...")
            print(f"   This will take ~{len(queries) * 0.15:.0f} seconds (~15 minutes)")
            print()
        else:
            queries = TEST_QUERIES

        stats = tester.benchmark_queries(queries, limit=5)
        tester.print_benchmark_stats(stats)

        # Save stats to file
        filename = 'benchmark_results_1000.json' if run_full_benchmark else 'benchmark_results.json'
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n💾 Results saved to: {filename}\n")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}\n")

    # 4. Test category filtering (skip in full benchmark)
    if not run_full_benchmark:
        try:
            tester.test_categories()
        except Exception as e:
            print(f"❌ Category test failed: {e}\n")

    print("✅ All tests completed!")
    if not run_full_benchmark:
        print("\n💡 Tip: Run 'python test_search_api.py --full' for 1000-query benchmark with percentiles")


if __name__ == "__main__":
    main()
