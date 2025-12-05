#!/usr/bin/env python3
"""
Complete system build: embeddings, mean, FAISS index
"""
import sys
import logging
from pathlib import Path
import numpy as np
import json

sys.path.append(str(Path(__file__).parent))

from config.config import DATA_DIR
from src.embedding_generator import EmbeddingGenerator
from src.vector_index import VectorIndex

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 70)
    logger.info("COMPLETE SYSTEM BUILD")
    logger.info("=" * 70)
    logger.info("")
    
    # Step 1: Generate embeddings (includes centering and mean calculation)
    logger.info("Step 1: Generating and centering embeddings...")
    generator = EmbeddingGenerator()
    generator.generate_all_embeddings(resume=True)
    logger.info("✅ Embeddings generated, centered, and mean vector saved")
    logger.info("")
    
    # Step 2: Build FAISS index (uses already-centered embeddings)
    logger.info("Step 2: Building FAISS index...")
    vector_index = VectorIndex()
    vector_index.build_index()
    logger.info("✅ FAISS index built")
    logger.info("")
    
    # Step 3: Verify
    logger.info("Step 3: Verification...")
    with open(DATA_DIR / "pmid_index.json", 'r') as f:
        pmids = json.load(f)
    
    embeddings = np.load(DATA_DIR / "embeddings.npy")
    logger.info(f"Embeddings: {len(embeddings):,}")
    logger.info(f"PMIDs: {len(pmids):,}")
    
    faiss_index = vector_index._load_index()
    if faiss_index:
        logger.info(f"FAISS index: {faiss_index.ntotal:,} vectors")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ SYSTEM BUILD COMPLETE!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()

