#!/usr/bin/env python3
"""
Delete old Qdrant collection using direct REST API calls.

Usage:
    export CONFIRM_DELETE=yes
    python delete_collection_simple.py
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
QDRANT_URL = os.environ["QDRANT_URL"]
OLD_COLLECTION_NAME = "synthetic_ads_optimized"
CONFIRM_DELETE = os.environ.get("CONFIRM_DELETE", "no").lower()


def main():
    print("="*80)
    print("🗑️  DELETE OLD QDRANT COLLECTION")
    print("="*80)
    print()

    # Check confirmation
    if CONFIRM_DELETE != "yes":
        print("⚠️  CONFIRM_DELETE environment variable not set to 'yes'")
        print()
        print("To delete the collection, set:")
        print("  export CONFIRM_DELETE=yes")
        print()
        sys.exit(1)

    # Test connection
    print(f"📡 Connecting to Qdrant at {QDRANT_URL}...")
    try:
        response = requests.get(f"{QDRANT_URL}/", timeout=10)
        response.raise_for_status()
        info = response.json()
        print(f"✅ Connected! Qdrant v{info.get('version', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        sys.exit(1)

    print()

    # List all collections
    print("📊 Current collections:")
    print()
    try:
        response = requests.get(f"{QDRANT_URL}/collections", timeout=30)
        response.raise_for_status()
        data = response.json()

        collections = data.get("result", {}).get("collections", [])

        if not collections:
            print("  No collections found.")
            sys.exit(0)

        collection_names = []
        for col in collections:
            col_name = col.get("name")
            collection_names.append(col_name)

            # Get collection info
            try:
                info_response = requests.get(
                    f"{QDRANT_URL}/collections/{col_name}",
                    timeout=30
                )
                info_response.raise_for_status()
                col_info = info_response.json().get("result", {})

                points = col_info.get("points_count", 0)
                vectors = col_info.get("vectors_count", 0)

                print(f"  📦 {col_name}")
                print(f"     Points: {points:,}")
                print(f"     Vectors: {vectors:,}")
                print()
            except Exception as e:
                print(f"  📦 {col_name} (couldn't get details)")
                print()

    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        sys.exit(1)

    # Check if old collection exists
    if OLD_COLLECTION_NAME not in collection_names:
        print(f"✅ Collection '{OLD_COLLECTION_NAME}' does not exist (already deleted?).")
        sys.exit(0)

    # Get info about the collection to delete
    print("="*80)
    print(f"⚠️  DELETING '{OLD_COLLECTION_NAME}'")
    print("="*80)
    print()

    try:
        response = requests.get(
            f"{QDRANT_URL}/collections/{OLD_COLLECTION_NAME}",
            timeout=30
        )
        response.raise_for_status()
        old_info = response.json().get("result", {})
        points = old_info.get("points_count", 0)
        vectors = old_info.get("vectors_count", 0)

        print(f"Deleting:")
        print(f"  - {points:,} points")
        print(f"  - {vectors:,} vectors")
        print(f"  - Will free ~400MB of memory")
        print()
    except Exception as e:
        print(f"⚠️  Couldn't get collection details: {e}")
        print()

    # Delete collection
    print(f"🗑️  Deleting '{OLD_COLLECTION_NAME}'...")

    try:
        response = requests.delete(
            f"{QDRANT_URL}/collections/{OLD_COLLECTION_NAME}",
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "ok":
            print(f"✅ Successfully deleted '{OLD_COLLECTION_NAME}'!")
            print()
            print(f"💾 Freed ~400MB of memory")
        else:
            print(f"❌ Unexpected response: {result}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error deleting collection: {e}")
        sys.exit(1)

    print()

    # List remaining collections
    print("📊 Remaining collections:")
    print()

    try:
        response = requests.get(f"{QDRANT_URL}/collections", timeout=30)
        response.raise_for_status()
        data = response.json()

        collections = data.get("result", {}).get("collections", [])

        if not collections:
            print("  No collections remaining.")
        else:
            for col in collections:
                col_name = col.get("name")

                # Get points count
                try:
                    info_response = requests.get(
                        f"{QDRANT_URL}/collections/{col_name}",
                        timeout=30
                    )
                    info_response.raise_for_status()
                    col_info = info_response.json().get("result", {})
                    points = col_info.get("points_count", 0)
                    print(f"  ✓ {col_name}: {points:,} points")
                except:
                    print(f"  ✓ {col_name}")

    except Exception as e:
        print(f"⚠️  Couldn't list remaining collections: {e}")

    print()
    print("="*80)
    print("✅ CLEANUP COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
