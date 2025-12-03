"""
Cross-encoder reranker for semantic search results.
Reranks FAISS candidates for improved precision.
"""
from sentence_transformers import CrossEncoder
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for search results."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder model.
        
        Args:
            model_name: HuggingFace model identifier
            
        Recommended models (speed vs accuracy tradeoff):
            - "cross-encoder/ms-marco-MiniLM-L-6-v2"  (fastest, good quality)
            - "cross-encoder/ms-marco-MiniLM-L-12-v2" (balanced)
            - "cross-encoder/ms-marco-electra-base"   (slower, highest quality)
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        logger.info("Cross-encoder loaded successfully")
    
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = None,
        text_field: str = "abstract",
        title_field: str = "title"
    ) -> list[dict]:
        """
        Rerank documents based on cross-encoder scores.
        
        Args:
            query: The search query
            documents: List of document dicts from FAISS search
            top_k: Number of results to return (None = return all, reranked)
            text_field: Key for abstract/text in document dict
            title_field: Key for title in document dict
            
        Returns:
            Reranked list of documents with added 'rerank_score' field
        """
        if not documents:
            return []
        
        # Build query-document pairs
        # Combine title + abstract for richer context
        pairs = []
        for doc in documents:
            title = doc.get(title_field, "")
            abstract = doc.get(text_field, "")
            
            # Combine title and abstract
            doc_text = f"{title}. {abstract}".strip()
            if not doc_text or doc_text == ".":
                doc_text = title or abstract or "No content available"
            
            pairs.append([query, doc_text])
        
        # Get cross-encoder scores
        logger.debug(f"Reranking {len(pairs)} documents")
        scores = self.model.predict(pairs)
        
        # Log score distribution for debugging
        scores_array = np.array(scores)
        logger.info(
            f"Rerank scores - min: {scores_array.min():.4f}, max: {scores_array.max():.4f}, "
            f"mean: {scores_array.mean():.4f}, std: {scores_array.std():.4f}"
        )
        
        # Warn if scores are compressed (indicates model may not be suitable for this domain)
        score_range = scores_array.max() - scores_array.min()
        if score_range < 0.01:
            logger.warning(
                f"Rerank score range is very narrow ({score_range:.4f}). "
                "Cross-encoder may not be differentiating well for this domain. "
                "FAISS scores will be used as tiebreaker."
            )
        
        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
            # Keep original FAISS score for reference
            if "similarity_score" in doc:
                doc["faiss_score"] = doc["similarity_score"]
            elif "score" in doc:
                doc["faiss_score"] = doc["score"]
        
        # Sort by rerank score (descending), with FAISS score as tiebreaker
        # This ensures that when rerank scores are similar (common with domain-specific corpora),
        # the original semantic similarity ranking is preserved
        reranked = sorted(
            documents,
            key=lambda x: (
                round(x["rerank_score"], 3),  # Round to 3 decimals to group similar scores
                x.get("faiss_score", x.get("similarity_score", 0))  # Use FAISS score as tiebreaker
            ),
            reverse=True
        )
        
        # IMPORTANT: Only filter if top_k is explicitly provided and less than total
        # This preserves recall - reranking should reorder, not filter
        if top_k is not None and top_k < len(reranked):
            # Only filter if we're explicitly limiting
            reranked = reranked[:top_k]
        
        return reranked


# Dictionary to store multiple reranker instances (one per model)
_reranker_instances = {}

# Available models
AVAILABLE_MODELS = {
    "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "best": "cross-encoder/ms-marco-electra-base",
    # Biomedical option - better for PubMed/medical literature
    # Note: This model may not exist yet, but placeholder for future use
    # "biomedical": "ncbi/MedCPT-Cross-Encoder"
}


def get_reranker(model_name: str = "best") -> Reranker:
    """
    Get or create a reranker instance for the specified model.
    Supports multiple models being loaded simultaneously.
    
    Args:
        model_name: Either a key from AVAILABLE_MODELS ("fast", "balanced", "best")
                   or a full HuggingFace model path
    
    Returns:
        Reranker instance for the specified model
    """
    global _reranker_instances
    
    # If model_name is a key, get the actual model path
    if model_name in AVAILABLE_MODELS:
        model_path = AVAILABLE_MODELS[model_name]
    else:
        # Assume it's already a full model path
        model_path = model_name
    
    # Return existing instance if available
    if model_path in _reranker_instances:
        return _reranker_instances[model_path]
    
    # Create new instance
    logger.info(f"Creating new reranker instance for: {model_path}")
    _reranker_instances[model_path] = Reranker(model_path)
    return _reranker_instances[model_path]

