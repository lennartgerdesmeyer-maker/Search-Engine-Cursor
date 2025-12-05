# Semantic Search Improvements

## Problem Identified

**Example Issue:**
- Query: "Prospective randomized trial directly comparing surgical repair versus nonoperative functional treatment for acute Achilles tendon rupture..."
- Expected Result: "Operative versus non-operative treatment of acute rupture of tendo Achillis: a prospective randomised evaluation of functional outcome"
- **Actual Rank: #1929** ❌ (Should be top 5!)

This indicates severe semantic search performance issues.

---

## Root Causes

### 1. **Query Centering Was Too Aggressive**
- Query embeddings were centered by subtracting the mean vector
- This removes "medical domain" components
- **Problem:** When queries ARE very medical/domain-specific, this REDUCES similarity scores
- **Result:** Relevant papers get low scores

### 2. **Missing MeSH Terms in Embeddings**
- Only used Title + Abstract
- MeSH terms contain crucial medical keywords
- **Result:** Missing important semantic signals

### 3. **Similarity Threshold Too Low**
- Was set to 0.05 (5%)
- Allowed very weak matches through
- **Result:** Results flooded with low-quality matches

---

## Improvements Implemented

### ✅ Fix #1: Query Centering Now Optional
**File:** `config/config.py`
**Change:**
```python
ENABLE_QUERY_CENTERING = False  # Disabled by default
```

**Impact:**
- Query embeddings now match against documents without aggressive normalization
- Should significantly improve similarity scores for domain-specific queries
- Can be re-enabled if needed by setting to `True`

---

### ✅ Fix #2: MeSH Terms Included in Embeddings
**File:** `src/embedding_generator.py`
**Change:**
```python
def create_embedding_text(self, row: pd.Series) -> str:
    # Now combines: Title + Abstract + MeSH Terms
    # MeSH terms provide domain keywords that improve matching
```

**Impact:**
- Embeddings now capture medical terminology from MeSH terms
- Better semantic matching for medical concepts
- **REQUIRES REGENERATING EMBEDDINGS** (see steps below)

---

### ✅ Fix #3: Increased Similarity Threshold
**File:** `config/config.py`
**Change:**
```python
SIMILARITY_THRESHOLD = 0.15  # Increased from 0.05
```

**Impact:**
- Filters out very weak matches
- With centering disabled, scores are naturally higher
- Better precision without sacrificing recall

---

## How to Apply These Improvements

### **IMPORTANT: Embeddings Must Be Regenerated**

Since we changed how document text is prepared (adding MeSH terms), you need to regenerate embeddings:

```bash
# Step 1: Backup existing embeddings (optional but recommended)
cp data/embeddings.npy data/embeddings_backup.npy
cp data/embedding_mean.npy data/embedding_mean_backup.npy
cp data/faiss_index.bin data/faiss_index_backup.bin

# Step 2: Regenerate embeddings with new settings
python src/embedding_generator.py

# Step 3: Rebuild FAISS index
python src/vector_index.py

# Step 4: Test the improvements
python app.py
```

### **Time Estimate:**
- Generating embeddings: ~1-3 hours (depending on corpus size)
- Building FAISS index: ~1-5 minutes
- Total: ~1-3 hours

---

## Testing the Improvements

### Test the Same Query Again:

```python
from src.search_engine import SemanticSearchEngine

engine = SemanticSearchEngine()

query = """Prospective randomized trial directly comparing surgical repair
versus nonoperative functional treatment for acute Achilles tendon rupture,
evaluating rerupture, complications, calf strength, and functional recovery"""

results = engine.search(query=query, top_k=50)

# Check if the target paper appears in top 10
for i, result in enumerate(results['results'][:10], 1):
    print(f"{i}. [{result['similarity_score']*100:.1f}%] {result['title']}")
```

### Expected Improvement:
- **Before:** Rank #1929, score ~30%
- **After:** Should be in **top 10**, score >50%

---

## Additional Improvements to Consider

### Quick Wins (Can Implement Now):

1. **Try a Better Model**
   ```python
   # In config.py, try:
   EMBEDDING_MODEL = "allenai/specter2"  # Specifically for scientific papers
   # OR
   EMBEDDING_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
   ```

2. **Enable Reranker by Default**
   - Reranker uses cross-encoder for much better precision
   - Set `use_reranker=True` in search calls
   - Only adds ~1-2 seconds for top_k=100

3. **Query Preprocessing**
   - Simplify long queries to key concepts
   - "surgical vs nonsurgical Achilles rupture RCT" may work better than long queries

---

### Future Improvements:

4. **Hybrid Search (BM25 + Semantic)**
   ```python
   final_score = 0.3 * bm25_score + 0.7 * semantic_score
   ```

5. **Fine-tune Model on Your Domain**
   - Use your labeled foot & ankle papers
   - Can improve performance by 10-20%

6. **Query Expansion**
   - Expand medical terms with synonyms
   - "Achilles tendon rupture" → "Achilles tear calcaneal tendon injury"

---

## Configuration Reference

### Current Settings (config/config.py):

```python
# Embedding Model
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBEDDING_DIMENSION = 768

# Search Settings
DEFAULT_TOP_K = 1000
MAX_TOP_K = 10000
SIMILARITY_THRESHOLD = 0.15  # Increased from 0.05
ENABLE_QUERY_CENTERING = False  # Disabled for better results

# Validation Mode (for testing against known studies)
VALIDATION_MODE = {
    "min_similarity": 0.0,
    "top_k": 5000,
    "use_reranker": False,
    "faiss_candidates": 10000
}
```

---

## Monitoring Performance

### Check Logs:
```bash
tail -f logs/search_engine.log
```

Look for:
- "Query centering DISABLED" (confirms new setting)
- Score ranges in search results
- Ranking of known relevant papers

### Compare Before/After:
```python
# Save test queries and expected papers
test_cases = [
    {
        "query": "surgical vs nonsurgical Achilles rupture",
        "expected_pmid": "12345678",
        "expected_title": "Operative versus non-operative..."
    }
]

# Run searches and log ranks
```

---

## Rollback Plan

If improvements don't work:

```bash
# Restore old embeddings
cp data/embeddings_backup.npy data/embeddings.npy
cp data/embedding_mean_backup.npy data/embedding_mean.npy
cp data/faiss_index_backup.bin data/faiss_index.bin

# Re-enable centering in config.py
ENABLE_QUERY_CENTERING = True
SIMILARITY_THRESHOLD = 0.05
```

---

## Questions?

If you encounter issues:
1. Check logs in `logs/search_engine.log`
2. Verify embeddings regenerated successfully
3. Test with simple queries first
4. Try adjusting SIMILARITY_THRESHOLD (0.10 - 0.20 range)

---

## Summary

**Changes Made:**
✅ Disabled query centering (configurable)
✅ Added MeSH terms to embeddings
✅ Increased similarity threshold
✅ Added detailed logging

**Next Steps:**
1. Regenerate embeddings (REQUIRED)
2. Test with your problematic query
3. Monitor performance and adjust if needed

**Expected Result:**
Your example query should now rank the target paper in the **top 10** instead of #1929!
