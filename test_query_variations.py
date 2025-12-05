"""
Test different query formulations to find the issue
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.search_engine import SemanticSearchEngine

def test_query_lengths():
    """Test how query length affects ranking"""

    engine = SemanticSearchEngine()

    # Target paper we're looking for
    target_title = "Operative versus non-operative treatment of acute rupture of tendo Achillis"

    queries = [
        ("ORIGINAL LONG", """Prospective randomized trial directly comparing surgical repair versus nonoperative functional treatment for acute Achilles tendon rupture, evaluating rerupture, complications, calf strength, and functional recovery; excluding adjunct therapies such as PRP or membrane augmentation."""),

        ("SHORT FOCUSED", "surgical versus nonsurgical treatment acute Achilles tendon rupture randomized trial"),

        ("VERY SHORT", "Achilles tendon rupture operative nonoperative RCT"),

        ("TITLE ONLY", "operative versus non-operative treatment acute rupture achilles tendon"),

        ("KEYWORDS", "Achilles rupture surgery conservative treatment prospective randomized"),
    ]

    print("=" * 80)
    print("TESTING QUERY LENGTH IMPACT")
    print("=" * 80)

    results_summary = []

    for query_name, query in queries:
        print(f"\n{'-' * 80}")
        print(f"Query: {query_name}")
        print(f"Text: {query[:80]}...")
        print(f"Length: {len(query)} chars, {len(query.split())} words")
        print(f"{'-' * 80}")

        results = engine.search(
            query=query,
            top_k=2500,
            min_similarity=0.0
        )

        # Find target paper
        found_rank = None
        found_score = None
        for i, result in enumerate(results['results'], 1):
            if target_title.lower() in result['title'].lower():
                found_rank = i
                found_score = result['similarity_score']
                print(f"\n✓ FOUND at rank #{found_rank}")
                print(f"  Score: {found_score:.4f} ({found_score*100:.1f}%)")
                print(f"  Title: {result['title']}")
                break

        if not found_rank:
            print(f"\n✗ NOT FOUND in top {len(results['results'])}")

        # Show top 3
        print(f"\nTop 3 results:")
        for i, result in enumerate(results['results'][:3], 1):
            score = result['similarity_score']
            print(f"  {i}. [{score:.4f} = {score*100:.1f}%] {result['title'][:70]}...")

        results_summary.append({
            'query_name': query_name,
            'query_length': len(query),
            'word_count': len(query.split()),
            'rank': found_rank,
            'score': found_score
        })

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Query Length vs Performance")
    print("=" * 80)
    print(f"{'Query Type':<20} {'Length':<10} {'Words':<10} {'Rank':<10} {'Score':<10}")
    print("-" * 80)

    for r in results_summary:
        rank_str = f"#{r['rank']}" if r['rank'] else "NOT FOUND"
        score_str = f"{r['score']*100:.1f}%" if r['score'] else "N/A"
        print(f"{r['query_name']:<20} {r['query_length']:<10} {r['word_count']:<10} {rank_str:<10} {score_str:<10}")

    # Find best
    valid_results = [r for r in results_summary if r['rank'] is not None]
    if valid_results:
        best = min(valid_results, key=lambda x: x['rank'])
        print(f"\n✓ BEST: {best['query_name']} → Rank #{best['rank']} (Score: {best['score']*100:.1f}%)")
        print(f"  → Shorter queries perform better!")

    print("\n" + "=" * 80)


def test_with_reranker():
    """Test if reranker improves results"""

    engine = SemanticSearchEngine()
    target_title = "Operative versus non-operative treatment of acute rupture of tendo Achillis"

    query = "surgical versus nonsurgical treatment acute Achilles tendon rupture randomized trial"

    print("\n" + "=" * 80)
    print("TESTING RERANKER IMPACT")
    print("=" * 80)
    print(f"Query: {query}")

    # Without reranker
    print(f"\n{'-' * 80}")
    print("WITHOUT RERANKER")
    print(f"{'-' * 80}")

    results_no_rerank = engine.search(
        query=query,
        top_k=100,
        use_reranker=False
    )

    rank_no_rerank = None
    for i, result in enumerate(results_no_rerank['results'], 1):
        if target_title.lower() in result['title'].lower():
            rank_no_rerank = i
            print(f"Target paper at rank #{i}")
            print(f"Score: {result['similarity_score']:.4f} ({result['similarity_score']*100:.1f}%)")
            break

    if not rank_no_rerank:
        print("Target paper NOT in top 100")

    # With reranker
    print(f"\n{'-' * 80}")
    print("WITH RERANKER")
    print(f"{'-' * 80}")

    results_with_rerank = engine.search(
        query=query,
        top_k=100,
        use_reranker=True,
        reranker_model="best",
        faiss_candidates=500
    )

    rank_with_rerank = None
    for i, result in enumerate(results_with_rerank['results'], 1):
        if target_title.lower() in result['title'].lower():
            rank_with_rerank = i
            print(f"Target paper at rank #{i}")
            print(f"Rerank score: {result.get('rerank_score', 'N/A')}")
            print(f"FAISS score: {result.get('similarity_score', 'N/A')}")
            break

    if not rank_with_rerank:
        print("Target paper NOT in top 100")

    # Summary
    print(f"\n{'-' * 80}")
    print("RERANKER COMPARISON")
    print(f"{'-' * 80}")
    if rank_no_rerank and rank_with_rerank:
        improvement = rank_no_rerank - rank_with_rerank
        if improvement > 0:
            print(f"✓ Reranker IMPROVED ranking by {improvement} positions")
            print(f"  Without reranker: #{rank_no_rerank}")
            print(f"  With reranker: #{rank_with_rerank}")
        else:
            print(f"✗ Reranker did not improve (or made it worse by {-improvement})")

    print("=" * 80)


if __name__ == "__main__":
    # Test 1: Query length impact
    test_query_lengths()

    # Test 2: Reranker impact
    test_with_reranker()
