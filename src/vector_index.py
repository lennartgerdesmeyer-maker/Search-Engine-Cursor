"""
Vector Index
FAISS-based vector index for fast similarity search
"""
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    DATA_DIR, LOG_DIR, EMBEDDING_DIMENSION,
    DEFAULT_TOP_K, MAX_TOP_K, SIMILARITY_THRESHOLD
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'vector_index.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VectorIndex:
    """FAISS-based vector index for semantic search"""
    
    def __init__(self):
        self.embeddings_path = DATA_DIR / "embeddings.npy"
        self.pmid_index_path = DATA_DIR / "pmid_index.json"
        self.faiss_index_path = DATA_DIR / "faiss_index.bin"
        
        self.index = None
        self.pmid_list = None
        self.embeddings = None
    
    def build_index(self, force_rebuild: bool = False):
        """Build or load the FAISS index"""
        
        if not force_rebuild and self.faiss_index_path.exists():
            logger.info("Loading existing FAISS index...")
            self.load_index()
            return
        
        logger.info("Building new FAISS index...")
        
        if not self.embeddings_path.exists():
            raise FileNotFoundError("Embeddings not found. Run embedding_generator.py first.")
        
        self.embeddings = np.load(self.embeddings_path).astype('float32')
        
        # Load PMID index with retry logic
        max_retries = 3
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                with open(self.pmid_index_path, 'r', encoding='utf-8') as f:
                    self.pmid_list = json.load(f)
                break
            except (IOError, OSError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Error loading PMID index (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to load PMID index after {max_retries} attempts: {e}")
                    raise
        
        logger.info(f"Loaded {len(self.pmid_list):,} embeddings")
        
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        
        batch_size = 10000
        for i in range(0, len(self.embeddings), batch_size):
            batch = self.embeddings[i:i + batch_size]
            self.index.add(batch)
        
        logger.info(f"Index built with {self.index.ntotal:,} vectors")
        
        faiss.write_index(self.index, str(self.faiss_index_path))
        logger.info(f"Index saved to {self.faiss_index_path}")
    
    def load_index(self):
        """Load existing index from disk"""
        if not self.faiss_index_path.exists():
            raise FileNotFoundError("FAISS index not found. Run build_index() first.")
        
        self.index = faiss.read_index(str(self.faiss_index_path))
        
        # Load PMID index with retry logic to handle potential I/O issues
        max_retries = 3
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                with open(self.pmid_index_path, 'r', encoding='utf-8') as f:
                    self.pmid_list = json.load(f)
                break
            except (IOError, OSError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Error loading PMID index (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to load PMID index after {max_retries} attempts: {e}")
                    raise
        
        logger.info(f"Loaded index with {self.index.ntotal:,} vectors")
    
    def search(self, query_embedding: np.ndarray, top_k: int = DEFAULT_TOP_K,
               min_similarity: float = SIMILARITY_THRESHOLD) -> List[Tuple[str, float]]:
        """Search for similar articles"""
        if self.index is None:
            self.load_index()
        
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Log query embedding info for debugging
        logger.debug(f"FAISS search called: query_embedding shape={query_embedding.shape}, "
                     f"norm={np.linalg.norm(query_embedding):.4f}, "
                     f"first_5_values={query_embedding[0, :5]}, "
                     f"top_k={top_k}, min_similarity={min_similarity}")
        
        top_k = min(top_k, MAX_TOP_K, self.index.ntotal)
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Log search results
        logger.debug(f"FAISS search returned: {len(similarities[0])} results, "
                     f"score_range=[{similarities[0].min():.4f}, {similarities[0].max():.4f}], "
                     f"first_5_indices={indices[0, :5]}, first_5_scores={similarities[0, :5]}")
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= min_similarity and idx < len(self.pmid_list):
                results.append((self.pmid_list[idx], float(sim)))
        
        logger.info(f"FAISS search: {len(results)} results after filtering (min_similarity={min_similarity})")
        if results:
            logger.debug(f"Top result: PMID={results[0][0]}, score={results[0][1]:.4f}")
            logger.debug(f"Bottom result: PMID={results[-1][0]}, score={results[-1][1]:.4f}")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = {
            "index_exists": self.faiss_index_path.exists(),
            "total_vectors": 0,
            "dimension": EMBEDDING_DIMENSION,
            "index_size_mb": 0
        }
        
        if self.faiss_index_path.exists():
            stats["index_size_mb"] = self.faiss_index_path.stat().st_size / (1024 * 1024)
            
            if self.index is None:
                self.load_index()
            stats["total_vectors"] = self.index.ntotal
        
        return stats


if __name__ == "__main__":
    index = VectorIndex()
    
    print("\n" + "="*60)
    print("VECTOR INDEX BUILDER")
    print("="*60)
    
    stats = index.get_stats()
    print(f"\nIndex exists: {stats['index_exists']}")
    print(f"Total vectors: {stats['total_vectors']:,}")
    
    response = input("\nBuild/rebuild index? (yes/no): ").strip().lower()
    if response == "yes":
        index.build_index(force_rebuild=True)
        
        stats = index.get_stats()
        print(f"\nIndex built: {stats['total_vectors']:,} vectors")
        print(f"Index size: {stats['index_size_mb']:.1f} MB")

