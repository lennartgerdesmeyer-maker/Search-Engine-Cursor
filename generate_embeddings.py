#!/usr/bin/env python3
"""
Clean script to generate all embeddings from scratch
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.embedding_generator import EmbeddingGenerator
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Starting embedding generation from scratch")
    logger.info("=" * 70)
    logger.info("")
    
    generator = EmbeddingGenerator()
    generator.generate_all_embeddings(resume=True)

