"""
Semantic Search Engine
Main search interface combining embeddings, vector index, and metadata
"""
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    DATA_DIR, LOG_DIR, METADATA_DB_PATH, EMBEDDING_MODEL,
    DEFAULT_TOP_K, SIMILARITY_THRESHOLD, EXPORT_DIR
)
from src.vector_index import VectorIndex
from src.reranker import get_reranker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'search_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """Main semantic search engine"""
    
    def __init__(self):
        self.model = None
        self.vector_index = VectorIndex()
        self.db_path = METADATA_DB_PATH
        self.last_query_embedding = None  # Store for analysis
        self.embedding_mean = None  # Mean vector for centering
        self._initialized = False  # Lazy initialization flag
    
    def _ensure_initialized(self):
        """Lazy initialization - only load index and mean when actually needed"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing search engine (loading FAISS index and mean vector)...")
            self.vector_index.build_index()
            # Load embedding mean for query centering
            mean_path = DATA_DIR / "embedding_mean.npy"
            if mean_path.exists():
                self.embedding_mean = np.load(mean_path)
                logger.info("Loaded embedding mean for query centering")
            else:
                logger.warning("Embedding mean not found - query centering disabled")
                self.embedding_mean = None
            self._initialized = True
            logger.info("Search engine initialized successfully")
        except Exception as e:
            logger.error(f"Error building/loading vector index: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _load_model(self):
        """Load embedding model for query encoding"""
        if self.model is None:
            logger.info(f"Loading model: {EMBEDDING_MODEL}")
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Model loaded")
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a search query into an embedding vector"""
        self._load_model()
        
        # Log the query being encoded
        logger.info(f"Encoding query: '{query[:100]}...' (length: {len(query)})")
        
        # Encode with normalization
        embedding = self.model.encode(query, normalize_embeddings=True)
        
        # Log embedding stats for debugging
        logger.debug(f"Query embedding after model encode - shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}, "
                    f"min: {embedding.min():.4f}, max: {embedding.max():.4f}, mean: {embedding.mean():.4f}")
        
        # Double-check normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Center and renormalize using the embedding mean
        # ENABLED - Query centering matches how document embeddings are stored
        # Both queries and documents are centered to remove common "medical domain" component
        # This improves similarity score differentiation
        if self.embedding_mean is not None:
            embedding_before_center = embedding.copy()
            similarity_to_mean = np.dot(embedding, self.embedding_mean)
            logger.debug(f"Query centering: similarity to mean before centering: {similarity_to_mean:.4f}")
            
            # Center query embedding (subtract mean)
            embedding = embedding - self.embedding_mean
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            logger.debug(f"Query embedding after centering - norm: {norm:.4f}, "
                        f"min: {embedding.min():.4f}, max: {embedding.max():.4f}, mean: {embedding.mean():.4f}")
            logger.debug("Query centering enabled - matches centered document embeddings")
        
        logger.debug(f"Final query embedding - norm: {np.linalg.norm(embedding):.4f}, "
                    f"min: {embedding.min():.4f}, max: {embedding.max():.4f}, mean: {embedding.mean():.4f}")
        
        return embedding
    
    def get_article_metadata(self, pmids: List[str]) -> Dict[str, Dict]:
        """Fetch metadata for a list of PMIDs"""
        if not pmids:
            return {}
        
        conn = sqlite3.connect(self.db_path)
        placeholders = ','.join(['?' for _ in pmids])
        query = f"SELECT * FROM articles WHERE pmid IN ({placeholders})"
        
        df = pd.read_sql_query(query, conn, params=pmids)
        conn.close()
        
        metadata = {}
        for _, row in df.iterrows():
            metadata[row['pmid']] = row.to_dict()
        
        return metadata
    
    def _ensure_initialized(self):
        """Lazy initialization - only load index and mean when actually needed"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing search engine (loading FAISS index and mean vector)...")
            self.vector_index.build_index()
            # Load embedding mean for query centering
            mean_path = DATA_DIR / "embedding_mean.npy"
            if mean_path.exists():
                self.embedding_mean = np.load(mean_path)
                logger.info("Loaded embedding mean for query centering")
            else:
                logger.warning("Embedding mean not found - query centering disabled")
                self.embedding_mean = None
            self._initialized = True
            logger.info("Search engine initialized successfully")
        except Exception as e:
            logger.error(f"Error building/loading vector index: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = SIMILARITY_THRESHOLD,
        date_before: Optional[str] = None,
        date_after: Optional[str] = None,
        pub_types: Optional[List[str]] = None,
        include_pmids: Optional[List[str]] = None,
        use_reranker: bool = False,
        faiss_candidates: int = 100,
        reranker_model: str = "best",
        validation_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Perform semantic search with optional cross-encoder reranking.
        
        Args:
            query: Search query string
            top_k: Number of final results to return
            min_similarity: Minimum similarity threshold for FAISS results
            date_before: Filter results before this date
            date_after: Filter results after this date
            pub_types: Filter by publication types
            include_pmids: List of PMIDs to check for matches
            use_reranker: Whether to apply cross-encoder reranking
            faiss_candidates: Number of candidates to retrieve from FAISS when reranking
                             (should be > top_k when using reranker)
        """
        # Lazy initialization - only load FAISS index when first search is performed
        self._ensure_initialized()
        
        start_time = datetime.now()
        
        # Log the query being searched
        logger.info(f"Search called with query: '{query[:200]}...' (length: {len(query)})")
        
        # Apply validation mode settings if enabled
        if validation_mode:
            from config.config import VALIDATION_MODE
            min_similarity = VALIDATION_MODE["min_similarity"]
            top_k = VALIDATION_MODE["top_k"]
            use_reranker = VALIDATION_MODE["use_reranker"]
            faiss_candidates = VALIDATION_MODE["faiss_candidates"]
            logger.info("Validation mode enabled: using maximum recall settings")
        
        query_embedding = self.encode_query(query)
        
        # Create a fingerprint of the query embedding for comparison
        query_fingerprint = hash(tuple(query_embedding[:10]))  # Hash first 10 values
        logger.info(f"Query embedding fingerprint: {query_fingerprint}")
        
        # Log query embedding stats
        logger.info(f"Query embedding stats: shape={query_embedding.shape}, norm={np.linalg.norm(query_embedding):.4f}, "
                    f"min={query_embedding.min():.4f}, max={query_embedding.max():.4f}, "
                    f"mean={query_embedding.mean():.4f}, first_5_values={query_embedding[:5]}")
        
        # When reranking, automatically calculate candidates if not enough provided
        # This ensures we have enough candidates after filtering to return top_k results
        if use_reranker:
            # Automatically set to top_k * 5 if faiss_candidates is too low
            # This accounts for filtering (date, pub_types, etc.) that reduces candidates
            # Use a more aggressive multiplier to ensure we have enough after filtering
            min_candidates = max(top_k * 5, faiss_candidates)
            search_k = min(min_candidates, 10000)  # Increased from 5000 to 10000
            # Remove similarity threshold for initial retrieval when reranking
            # The reranker will handle precision, so we want more candidates
            effective_min_similarity = 0.0  # No threshold, let reranker handle precision
            logger.info(f"Reranking enabled: retrieving {search_k} candidates for top_k={top_k} (min_similarity: {effective_min_similarity})")
        else:
            search_k = min(top_k * 5, 10000)  # Increased multiplier from 3 to 5, limit from 5000 to 10000
            effective_min_similarity = min_similarity
        
        # Store query embedding for analysis (ensure it's 1D array)
        # encode_query returns 1D array, so we store it directly
        self.last_query_embedding = query_embedding.flatten().copy() if query_embedding.ndim > 1 else query_embedding.copy()
        
        raw_results = self.vector_index.search(query_embedding, search_k, effective_min_similarity)
        
        logger.info(f"FAISS returned {len(raw_results)} raw results (requested {search_k})")
        
        # Log score range for debugging
        if raw_results:
            scores_list = [r[1] for r in raw_results]
            logger.info(f"Score range: min={min(scores_list):.4f}, max={max(scores_list):.4f}, mean={sum(scores_list)/len(scores_list):.4f}")
            logger.info(f"First 5 scores: {[f'{s:.4f}' for s in scores_list[:5]]}")
            logger.info(f"Last 5 scores: {[f'{s:.4f}' for s in scores_list[-5:]]}")
            
            # Analyze missed papers if included PMIDs provided
            if include_pmids:
                self._analyze_missed_papers(include_pmids, raw_results, effective_min_similarity)
        
        if not raw_results:
            return {
                "query": query,
                "total_results": 0,
                "results": [],
                "search_time_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
        
        pmids = [r[0] for r in raw_results]
        scores = {r[0]: r[1] for r in raw_results}
        
        metadata = self.get_article_metadata(pmids)
        logger.info(f"Retrieved metadata for {len(metadata)} articles")
        
        filtered_results = []
        # Ensure all PMIDs are strings for consistent comparison
        include_pmids_set = set(str(pmid) for pmid in include_pmids) if include_pmids else None
        
        for pmid in pmids:
            if pmid not in metadata:
                continue
            
            article = metadata[pmid]
            
            if date_before and article.get('pub_date'):
                if article['pub_date'] > date_before:
                    continue
            
            if date_after and article.get('pub_date'):
                if article['pub_date'] < date_after:
                    continue
            
            if pub_types:
                article_pub_types = article.get('pub_type', '').lower()
                if not any(pt.lower() in article_pub_types for pt in pub_types):
                    continue
            
            result = {
                "pmid": pmid,
                "doi": article.get('doi', ''),
                "title": article.get('title', ''),
                "authors": article.get('authors', ''),
                "journal": article.get('journal', ''),
                "pub_date": article.get('pub_date', ''),
                "pub_year": article.get('pub_year'),
                "pub_type": article.get('pub_type', ''),
                "abstract": article.get('abstract', ''),
                "mesh_terms": article.get('mesh_terms', ''),
                "similarity_score": round(scores[pmid], 4),
                "is_match": str(pmid) in include_pmids_set if include_pmids_set else None
            }
            
            filtered_results.append(result)
            
            # Only break early if not using reranker (reranker needs all candidates)
            if not use_reranker and len(filtered_results) >= top_k:
                break
        
        logger.info(f"After filtering: {len(filtered_results)} candidates (requested top_k={top_k})")
        
        # Log similarity score range after filtering
        if filtered_results:
            sim_scores = [r.get('similarity_score', 0) for r in filtered_results]
            logger.info(f"Similarity scores after filtering - min={min(sim_scores):.4f}, max={max(sim_scores):.4f}, mean={sum(sim_scores)/len(sim_scores):.4f}")
            logger.info(f"First result score: {filtered_results[0].get('similarity_score', 0):.4f}")
            logger.info(f"Last result score: {filtered_results[-1].get('similarity_score', 0):.4f}")
        
        # Apply reranking if enabled
        if use_reranker and filtered_results:
            try:
                logger.info(f"Reranking {len(filtered_results)} candidates (will preserve all, just reorder)")
                reranker = get_reranker(reranker_model)
                # Don't pass top_k to reranker - let it return all reranked results
                # We'll limit at the end if needed
                filtered_results = reranker.rerank(
                    query=query,
                    documents=filtered_results,
                    top_k=None,  # Return all reranked results, don't filter
                    text_field="abstract",
                    title_field="title"
                )
                logger.info(f"Reranked {len(filtered_results)} results using {reranker_model} (preserved all)")
                # Now limit to top_k if needed (single filter point)
                if len(filtered_results) > top_k:
                    filtered_results = filtered_results[:top_k]
                    logger.info(f"Limited to top {top_k} after reranking")
            except Exception as e:
                logger.error(f"Error during reranking: {e}")
                # Fall back to original results if reranking fails
                filtered_results = filtered_results[:top_k]
        
        match_stats = None
        if include_pmids_set and len(include_pmids_set) > 0:
            # Ensure all PMIDs are strings for consistent comparison
            found_pmids = set(str(r['pmid']) for r in filtered_results)
            matched = found_pmids & include_pmids_set
            recall = len(matched) / len(include_pmids_set) if len(include_pmids_set) > 0 else 0.0
            match_stats = {
                "total_included_studies": len(include_pmids_set),
                "matched": len(matched),
                "missed": len(include_pmids_set - found_pmids),
                "new_candidates": len(found_pmids - include_pmids_set),
                "recall": round(recall, 4)  # Round to avoid floating point issues
            }
            logger.info(f"Match statistics: {match_stats}")
            logger.info(f"Matched PMIDs: {list(matched)[:10]}...")  # Log first 10 matched
            logger.info(f"Missed PMIDs: {list(include_pmids_set - found_pmids)[:10]}...")  # Log first 10 missed
        
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "query": query,
            "parameters": {
                "top_k": top_k,
                "min_similarity": min_similarity,
                "date_before": date_before,
                "date_after": date_after,
                "pub_types": pub_types,
                "use_reranker": use_reranker,
                "reranker_model": reranker_model if use_reranker else None,
                "faiss_candidates": faiss_candidates if use_reranker else None
            },
            "total_results": len(filtered_results),
            "results": filtered_results,
            "match_statistics": match_stats,
            "reranked": use_reranker,
            "search_time_ms": round(search_time, 2)
        }
    
    def apply_keyword_ranking(
        self,
        results: List[Dict[str, Any]],
        boost_terms: Optional[List[str]] = None,
        penalize_terms: Optional[List[str]] = None,
        boost_weight: float = 0.15,
        penalty_weight: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Adjust semantic similarity scores based on keyword preferences using diminishing returns.
        Papers with boost_terms get higher scores, penalize_terms get lower scores.
        
        Args:
            results: List of semantic search results with 'similarity_score' field
            boost_terms: Terms that should increase ranking (list of strings)
            penalize_terms: Terms that should decrease ranking (list of strings)
            boost_weight: Base boost amount (0.1-0.2 recommended)
            penalty_weight: Base penalty amount (0.1-0.2 recommended)
        
        Returns:
            Re-ranked results with adjusted scores and ranking notes
        """
        if not boost_terms and not penalize_terms:
            return results
        
        # Normalize terms to lowercase
        boost_terms_lower = [term.lower().strip() for term in boost_terms] if boost_terms else []
        penalize_terms_lower = [term.lower().strip() for term in penalize_terms] if penalize_terms else []
        
        # Remove empty terms
        boost_terms_lower = [t for t in boost_terms_lower if t]
        penalize_terms_lower = [t for t in penalize_terms_lower if t]
        
        if not boost_terms_lower and not penalize_terms_lower:
            return results
        
        adjusted_count = 0
        
        for result in results:
            # Combine title and abstract for searching
            title = (result.get('title') or '').lower()
            abstract = (result.get('abstract') or '').lower()
            text = f"{title} {abstract}"
            
            original_score = result.get('similarity_score', 0.0)
            adjusted_score = original_score
            
            # Track why score changed (for UI display)
            ranking_notes = []
            
            # Apply boosts with diminishing returns
            if boost_terms_lower:
                matched_boosts = []
                for term in boost_terms_lower:
                    if term in text:
                        matched_boosts.append(term)
                
                if matched_boosts:
                    # Diminishing returns: first match gets full boost, additional matches get less
                    # Formula: boost_weight * min(matches, 3) * 0.7^(max(0, matches-1))
                    match_count = len(matched_boosts)
                    boost_multiplier = min(match_count, 3) * (0.7 ** max(0, match_count - 1))
                    boost = boost_weight * boost_multiplier
                    
                    adjusted_score += boost
                    result['original_similarity_score'] = original_score
                    ranking_notes.append(f"↑ Boosted: contains {', '.join(matched_boosts[:3])}")
                    adjusted_count += 1
            
            # Apply penalties with diminishing returns
            if penalize_terms_lower:
                matched_penalties = []
                for term in penalize_terms_lower:
                    if term in text:
                        matched_penalties.append(term)
                
                if matched_penalties:
                    # Diminishing returns for penalties too
                    match_count = len(matched_penalties)
                    penalty_multiplier = min(match_count, 3) * (0.7 ** max(0, match_count - 1))
                    penalty = penalty_weight * penalty_multiplier
                    
                    adjusted_score -= penalty
                    if 'original_similarity_score' not in result:
                        result['original_similarity_score'] = original_score
                    ranking_notes.append(f"↓ Penalized: contains {', '.join(matched_penalties[:3])}")
                    if not ranking_notes or not ranking_notes[0].startswith('↑'):
                        adjusted_count += 1
            
            # Store ranking info
            if ranking_notes:
                result['similarity_score'] = adjusted_score
                result['ranking_notes'] = ranking_notes
        
        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.get('similarity_score', 0.0), reverse=True)
        
        logger.info(f"Applied keyword ranking: {adjusted_count} papers adjusted")
        
        return results
    
    def export_results_csv(self, results: Dict[str, Any], filename: str = None, screening_results: Dict[str, Any] = None) -> str:
        """Export search results to CSV with optional screening decision column"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"search_results_{timestamp}.csv"
        
        filepath = EXPORT_DIR / filename
        
        df = pd.DataFrame(results['results'])
        
        # Add screening decision column if screening results are available
        if screening_results and screening_results.get('results'):
            # Create a mapping of PMID to screening decision
            screening_map = {}
            for screening_result in screening_results['results']:
                pmid = str(screening_result.get('pmid', ''))
                final_category = screening_result.get('final_category', '')
                
                # Map final_category to display format
                if final_category == 'include':
                    decision = 'Included'
                elif final_category == 'exclude':
                    decision = 'Excluded'
                elif final_category in ['uncertain', 'disagreement', 'error']:
                    decision = 'Uncertain'
                else:
                    decision = 'Uncertain'
                
                screening_map[pmid] = decision
            
            # Add screening decision column
            df['Screening Decision'] = df['pmid'].astype(str).map(screening_map).fillna('Not Screened')
        else:
            # No screening results available
            df['Screening Decision'] = 'Not Screened'
        
        df.to_csv(filepath, index=False)
        
        logger.info(f"Results exported to {filepath}")
        return str(filepath)
    
    def export_results_ris(self, results: Dict[str, Any], filename: str = None) -> str:
        """Export search results to RIS format"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"search_results_{timestamp}.ris"
        
        filepath = EXPORT_DIR / filename
        
        ris_entries = []
        for article in results['results']:
            entry = []
            entry.append("TY  - JOUR")
            entry.append(f"TI  - {article.get('title', '')}")
            
            authors = article.get('authors', '').split('; ')
            for author in authors:
                if author:
                    entry.append(f"AU  - {author}")
            
            entry.append(f"JO  - {article.get('journal', '')}")
            entry.append(f"PY  - {article.get('pub_year', '')}")
            entry.append(f"DA  - {article.get('pub_date', '')}")
            
            if article.get('doi'):
                entry.append(f"DO  - {article.get('doi')}")
            
            entry.append(f"AN  - PMID:{article.get('pmid', '')}")
            
            if article.get('abstract'):
                entry.append(f"AB  - {article.get('abstract')}")
            
            mesh = article.get('mesh_terms', '').split('; ')
            for term in mesh:
                if term:
                    entry.append(f"KW  - {term}")
            
            entry.append("ER  - ")
            entry.append("")
            
            ris_entries.append("\n".join(entry))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(ris_entries))
        
        logger.info(f"Results exported to {filepath}")
        return str(filepath)
    
    def _analyze_missed_papers(self, included_pmids: List[str], raw_results: List[Tuple[str, float]], min_similarity: float):
        """Analyze why included papers were missed."""
        if not included_pmids or self.last_query_embedding is None:
            return
        
        result_pmids = {str(r[0]) for r in raw_results}
        missed_pmids = [str(pmid) for pmid in included_pmids if str(pmid) not in result_pmids]
        
        if not missed_pmids:
            logger.info("All included papers found in search results")
            return
        
        logger.warning(f"Analyzing {len(missed_pmids)} missed papers...")
        
        # Load embeddings if needed
        if self.vector_index.embeddings is None:
            self.vector_index.embeddings = np.load(self.vector_index.embeddings_path).astype('float32')
        
        # Create PMID to index mapping
        pmid_to_index = {pmid: idx for idx, pmid in enumerate(self.vector_index.pmid_list)}
        
        for pmid in missed_pmids[:10]:  # Analyze first 10 missed
            if pmid not in pmid_to_index:
                logger.warning(f"  PMID {pmid}: NOT IN DATABASE")
                continue
            
            # Get its embedding index
            idx = pmid_to_index[pmid]
            
            try:
                # Calculate its similarity to the query
                paper_embedding = self.vector_index.embeddings[idx]
                
                # Ensure both are 1D arrays with matching shapes
                query_vec = np.asarray(self.last_query_embedding).flatten()
                paper_vec = np.asarray(paper_embedding).flatten()
                
                # Check shapes match
                if query_vec.shape[0] != paper_vec.shape[0]:
                    logger.warning(f"  PMID {pmid}: Shape mismatch (query: {query_vec.shape}, paper: {paper_vec.shape}), skipping")
                    continue
                
                similarity = np.dot(query_vec, paper_vec)
                # Convert to Python float
                similarity = float(similarity.item() if hasattr(similarity, 'item') else similarity)
            except Exception as e:
                logger.warning(f"  PMID {pmid}: Error calculating similarity - {e}")
                continue
            
            # Determine reason for exclusion
            if similarity < min_similarity:
                logger.warning(f"  PMID {pmid}: similarity={similarity:.4f} ({similarity*100:.1f}%) - Excluded by threshold ({min_similarity})")
            else:
                # Find what the lowest score in results was
                min_result_score = min([r[1] for r in raw_results]) if raw_results else 0
                logger.warning(f"  PMID {pmid}: similarity={similarity:.4f} ({similarity*100:.1f}%) - Excluded by top_k limit (min result score: {min_result_score:.4f})")


def test_search():
    """Quick test of the search engine"""
    engine = SemanticSearchEngine()
    
    test_query = """
    What is the effectiveness of surgical versus conservative treatment 
    for acute Achilles tendon rupture?
    """
    
    results = engine.search(query=test_query, top_k=10, min_similarity=0.3)
    
    print(f"\nQuery: {test_query[:80]}...")
    print(f"Found: {results['total_results']} results")
    print(f"Search time: {results['search_time_ms']:.1f} ms")
    print("\nTop 5 results:")
    
    for i, r in enumerate(results['results'][:5], 1):
        print(f"\n{i}. [{r['similarity_score']:.3f}] {r['title'][:80]}...")
        print(f"   PMID: {r['pmid']} | {r['journal']} | {r['pub_year']}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SEMANTIC SEARCH ENGINE TEST")
    print("="*60)
    
    test_search()

