"""
Diagnostic Script: Test Semantic Search with and without Centering
Compare results to identify why relevant papers are ranked poorly
"""
import numpy as np
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent))
from src.search_engine import SemanticSearchEngine
from config.config import EMBEDDING_MODEL, DATA_DIR

def test_query_with_and_without_centering():
    """Test the specific problematic query with centering on/off"""

    # The problematic query
    query = """Prospective randomized trial directly comparing surgical repair versus nonoperative functional treatment for acute Achilles tendon rupture, evaluating rerupture, complications, calf strength, and functional recovery; excluding adjunct therapies such as PRP or membrane augmentation."""

    # Expected paper that should rank highly
    expected_title = "Operative versus non-operative treatment of acute rupture of tendo Achillis: a prospective randomised evaluation of functional outcome"

    print("=" * 80)
    print("DIAGNOSTIC TEST: Query Centering Impact")
    print("=" * 80)
    print(f"\nQuery: {query[:100]}...")
    print(f"\nExpected high-ranking paper: {expected_title}")
    print("\n" + "=" * 80)

    # Load the model
    print(f"\nLoading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Load embedding mean
    mean_path = DATA_DIR / "embedding_mean.npy"
    if mean_path.exists():
        embedding_mean = np.load(mean_path)
        print(f"Loaded embedding mean (shape: {embedding_mean.shape})")
    else:
        print("WARNING: No embedding mean found - centering disabled")
        embedding_mean = None

    # Encode query WITHOUT centering
    print("\n" + "-" * 80)
    print("TEST 1: Query embedding WITHOUT centering")
    print("-" * 80)
    embedding_no_center = model.encode(query, normalize_embeddings=True)
    norm_no_center = np.linalg.norm(embedding_no_center)
    if norm_no_center > 0:
        embedding_no_center = embedding_no_center / norm_no_center

    print(f"Shape: {embedding_no_center.shape}")
    print(f"Norm: {np.linalg.norm(embedding_no_center):.6f}")
    print(f"Mean: {embedding_no_center.mean():.6f}")
    print(f"Std: {embedding_no_center.std():.6f}")
    print(f"Min: {embedding_no_center.min():.6f}")
    print(f"Max: {embedding_no_center.max():.6f}")
    print(f"First 10 values: {embedding_no_center[:10]}")

    # Encode query WITH centering
    print("\n" + "-" * 80)
    print("TEST 2: Query embedding WITH centering")
    print("-" * 80)
    embedding_with_center = model.encode(query, normalize_embeddings=True)
    norm = np.linalg.norm(embedding_with_center)
    if norm > 0:
        embedding_with_center = embedding_with_center / norm

    if embedding_mean is not None:
        similarity_to_mean = np.dot(embedding_with_center, embedding_mean)
        print(f"Similarity to mean before centering: {similarity_to_mean:.6f}")

        # Center
        embedding_with_center = embedding_with_center - embedding_mean
        norm = np.linalg.norm(embedding_with_center)
        if norm > 0:
            embedding_with_center = embedding_with_center / norm

        print(f"After centering:")
        print(f"  Shape: {embedding_with_center.shape}")
        print(f"  Norm: {np.linalg.norm(embedding_with_center):.6f}")
        print(f"  Mean: {embedding_with_center.mean():.6f}")
        print(f"  Std: {embedding_with_center.std():.6f}")
        print(f"  Min: {embedding_with_center.min():.6f}")
        print(f"  Max: {embedding_with_center.max():.6f}")
        print(f"  First 10 values: {embedding_with_center[:10]}")

    # Calculate difference
    if embedding_mean is not None:
        diff = np.abs(embedding_no_center - embedding_with_center)
        print(f"\n  Embedding change: mean diff = {diff.mean():.6f}, max diff = {diff.max():.6f}")

    # Run actual searches
    print("\n" + "=" * 80)
    print("RUNNING ACTUAL SEARCHES")
    print("=" * 80)

    engine = SemanticSearchEngine()

    # Test WITHOUT centering (temporarily disable)
    print("\n" + "-" * 80)
    print("SEARCH 1: WITHOUT centering (temporarily disabled)")
    print("-" * 80)
    original_mean = engine.embedding_mean
    engine.embedding_mean = None  # Disable centering

    results_no_center = engine.search(
        query=query,
        top_k=2000,
        min_similarity=0.0  # No threshold to see all results
    )

    # Find the expected paper in results
    rank_no_center = None
    score_no_center = None
    for i, result in enumerate(results_no_center['results'], 1):
        if expected_title.lower() in result['title'].lower() or result['title'].lower() in expected_title.lower():
            rank_no_center = i
            score_no_center = result['similarity_score']
            print(f"\n✓ FOUND at rank #{rank_no_center}")
            print(f"  Similarity: {score_no_center:.4f} ({score_no_center*100:.1f}%)")
            print(f"  Title: {result['title']}")
            break

    if rank_no_center is None:
        print(f"\n✗ NOT FOUND in top {len(results_no_center['results'])} results")

    # Show top 5 results
    print(f"\nTop 5 results WITHOUT centering:")
    for i, result in enumerate(results_no_center['results'][:5], 1):
        print(f"  {i}. [{result['similarity_score']:.4f}] {result['title'][:80]}...")

    # Test WITH centering (restore original)
    print("\n" + "-" * 80)
    print("SEARCH 2: WITH centering (current implementation)")
    print("-" * 80)
    engine.embedding_mean = original_mean  # Re-enable centering

    results_with_center = engine.search(
        query=query,
        top_k=2000,
        min_similarity=0.0  # No threshold
    )

    # Find the expected paper
    rank_with_center = None
    score_with_center = None
    for i, result in enumerate(results_with_center['results'], 1):
        if expected_title.lower() in result['title'].lower() or result['title'].lower() in expected_title.lower():
            rank_with_center = i
            score_with_center = result['similarity_score']
            print(f"\n✓ FOUND at rank #{rank_with_center}")
            print(f"  Similarity: {score_with_center:.4f} ({score_with_center*100:.1f}%)")
            print(f"  Title: {result['title']}")
            break

    if rank_with_center is None:
        print(f"\n✗ NOT FOUND in top {len(results_with_center['results'])} results")

    # Show top 5 results
    print(f"\nTop 5 results WITH centering:")
    for i, result in enumerate(results_with_center['results'][:5], 1):
        print(f"  {i}. [{result['similarity_score']:.4f}] {result['title'][:80]}...")

    # Summary comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    if rank_no_center and rank_with_center:
        print(f"\nTarget paper rank:")
        print(f"  WITHOUT centering: #{rank_no_center} (score: {score_no_center:.4f} = {score_no_center*100:.1f}%)")
        print(f"  WITH centering:    #{rank_with_center} (score: {score_with_center:.4f} = {score_with_center*100:.1f}%)")

        if rank_no_center < rank_with_center:
            improvement = rank_with_center - rank_no_center
            print(f"\n✓ CENTERING HURTS PERFORMANCE by {improvement} ranks!")
            print(f"  Recommendation: DISABLE CENTERING")
        elif rank_no_center > rank_with_center:
            improvement = rank_no_center - rank_with_center
            print(f"\n✓ CENTERING HELPS PERFORMANCE by {improvement} ranks!")
            print(f"  Recommendation: KEEP CENTERING")
        else:
            print(f"\n→ NO DIFFERENCE")

    print("\n" + "=" * 80)


def test_query_variations():
    """Test different query formulations"""
    print("\n" + "=" * 80)
    print("TESTING QUERY VARIATIONS")
    print("=" * 80)

    queries = [
        # Original long query
        ("LONG (original)", """Prospective randomized trial directly comparing surgical repair versus nonoperative functional treatment for acute Achilles tendon rupture, evaluating rerupture, complications, calf strength, and functional recovery; excluding adjunct therapies such as PRP or membrane augmentation."""),

        # Short focused query
        ("SHORT", "surgical versus nonsurgical treatment acute Achilles tendon rupture randomized trial"),

        # Keywords only
        ("KEYWORDS", "Achilles tendon rupture operative nonoperative RCT prospective"),

        # Natural language
        ("NATURAL", "What is the effectiveness of surgical versus conservative treatment for acute Achilles tendon rupture?"),
    ]

    expected_title = "Operative versus non-operative treatment of acute rupture of tendo Achillis"

    engine = SemanticSearchEngine()

    results_summary = []

    for query_type, query in queries:
        print(f"\n{'-' * 80}")
        print(f"Query type: {query_type}")
        print(f"Query: {query[:100]}...")
        print(f"{'-' * 80}")

        results = engine.search(
            query=query,
            top_k=2000,
            min_similarity=0.0
        )

        # Find target paper
        rank = None
        score = None
        for i, result in enumerate(results['results'], 1):
            if expected_title.lower() in result['title'].lower():
                rank = i
                score = result['similarity_score']
                break

        if rank:
            print(f"✓ Target paper rank: #{rank} (score: {score:.4f} = {score*100:.1f}%)")
            results_summary.append((query_type, rank, score))
        else:
            print(f"✗ Target paper NOT FOUND in top 2000")
            results_summary.append((query_type, None, None))

        # Show top 3
        print(f"\nTop 3 results:")
        for i, result in enumerate(results['results'][:3], 1):
            print(f"  {i}. [{result['similarity_score']:.4f}] {result['title'][:80]}...")

    # Summary
    print("\n" + "=" * 80)
    print("QUERY VARIATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Query Type':<15} {'Rank':<10} {'Score':<10}")
    print("-" * 40)
    for query_type, rank, score in results_summary:
        rank_str = f"#{rank}" if rank else "NOT FOUND"
        score_str = f"{score*100:.1f}%" if score else "N/A"
        print(f"{query_type:<15} {rank_str:<10} {score_str:<10}")

    best = min(results_summary, key=lambda x: x[1] if x[1] else float('inf'))
    print(f"\nBest performing query type: {best[0]} (rank #{best[1]})")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SEMANTIC SEARCH DIAGNOSTIC TOOL")
    print("=" * 80)

    # Test 1: Centering impact
    test_query_with_and_without_centering()

    # Test 2: Query variations
    test_query_variations()

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
