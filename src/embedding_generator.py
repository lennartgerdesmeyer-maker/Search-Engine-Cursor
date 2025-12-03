"""
Embedding Generator
Creates vector embeddings for all articles using PubMedBERT
"""
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    DATA_DIR, LOG_DIR, METADATA_DB_PATH,
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, SAVE_CHECKPOINT_EVERY
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'embedding_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates and stores embeddings for article abstracts"""
    
    def __init__(self):
        self.model = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.embeddings_path = DATA_DIR / "embeddings.npy"
        self.pmid_index_path = DATA_DIR / "pmid_index.json"
        self.checkpoint_path = DATA_DIR / "embedding_checkpoint.json"
        
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            logger.info(f"Loading model: {EMBEDDING_MODEL}")
            logger.info("This may take a few minutes on first run...")
            
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.model.to(self.device)
            
            logger.info("Model loaded successfully")
    
    def get_articles_for_embedding(self) -> pd.DataFrame:
        """Get all articles for embedding (prefer abstracts, use titles if no abstract)"""
        logger.info("Loading articles from database...")
        
        conn = sqlite3.connect(METADATA_DB_PATH)
        # Get all articles, not just those with abstracts
        df = pd.read_sql_query(
            "SELECT pmid, title, abstract FROM articles",
            conn
        )
        conn.close()
        
        logger.info(f"Loaded {len(df):,} articles")
        return df
    
    def create_embedding_text(self, row: pd.Series) -> str:
        """Create the text to embed (title + abstract, or just title if no abstract)"""
        title = row['title'] if pd.notna(row['title']) else ""
        abstract = row['abstract'] if pd.notna(row['abstract']) and row['abstract'] != '' else ""
        
        if abstract:
            return f"{title} {abstract}".strip()
        else:
            # Use just the title if no abstract
            return title.strip()
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        self.load_model()
        
        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def save_checkpoint(self, processed: int, total: int):
        """Save progress checkpoint"""
        checkpoint = {
            "processed": processed,
            "total": total,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
    
    def load_checkpoint(self) -> Optional[int]:
        """Load previous checkpoint"""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            return checkpoint.get("processed", 0)
        return 0
    
    def generate_all_embeddings(self, resume: bool = True):
        """Generate embeddings for all articles"""
        logger.info("=" * 60)
        logger.info("STARTING EMBEDDING GENERATION")
        logger.info("=" * 60)
        
        df = self.get_articles_for_embedding()
        
        if len(df) == 0:
            logger.error("No articles found. Run pubmed_downloader.py first.")
            return
        
        start_idx = 0
        existing_embeddings = None
        existing_pmids = []
        
        if resume and self.embeddings_path.exists():
            if self.pmid_index_path.exists():
                with open(self.pmid_index_path, 'r') as f:
                    existing_pmids = json.load(f)
                existing_embeddings = np.load(self.embeddings_path)
                logger.info(f"Found {len(existing_pmids):,} existing embeddings")
            else:
                # Fallback to old checkpoint method
                start_idx = self.load_checkpoint()
                if start_idx > 0:
                    logger.info(f"Resuming from checkpoint: {start_idx:,} embeddings")
        
        df['embedding_text'] = df.apply(self.create_embedding_text, axis=1)
        
        # Filter out articles that are already embedded (by PMID, not position)
        if existing_pmids:
            remaining_df = df[~df['pmid'].isin(existing_pmids)]
            logger.info(f"Skipping {len(df) - len(remaining_df):,} already embedded articles")
        else:
            remaining_df = df.iloc[start_idx:]
        
        if len(remaining_df) == 0:
            logger.info("All embeddings already generated!")
            return
        
        logger.info(f"Generating embeddings for {len(remaining_df):,} articles...")
        
        self.load_model()
        
        chunk_size = SAVE_CHECKPOINT_EVERY
        all_new_embeddings = []
        all_new_pmids = []
        
        for chunk_start in tqdm(range(0, len(remaining_df), chunk_size), desc="Processing"):
            chunk_end = min(chunk_start + chunk_size, len(remaining_df))
            chunk_df = remaining_df.iloc[chunk_start:chunk_end]
            
            texts = chunk_df['embedding_text'].tolist()
            chunk_embeddings = self.generate_embeddings(texts)
            
            all_new_embeddings.append(chunk_embeddings)
            all_new_pmids.extend(chunk_df['pmid'].tolist())
            
            processed = len(existing_pmids) + len(all_new_pmids)
            self.save_checkpoint(processed, len(df))
            
            if (chunk_end % (chunk_size * 2)) == 0 or chunk_end == len(remaining_df):
                self._save_embeddings(
                    all_new_embeddings, all_new_pmids,
                    existing_embeddings, existing_pmids
                )
                logger.info(f"Saved checkpoint at {processed:,} articles")
        
        self._save_embeddings(
            all_new_embeddings, all_new_pmids,
            existing_embeddings, existing_pmids
        )
        
        logger.info("=" * 60)
        logger.info(f"EMBEDDING GENERATION COMPLETE")
        logger.info(f"Total: {len(existing_pmids) + len(all_new_pmids):,}")
        logger.info("=" * 60)
    
    def _save_embeddings(self, new_embeddings, new_pmids, existing_embeddings, existing_pmids):
        """Save embeddings to disk"""
        if new_embeddings:
            combined_new = np.vstack(new_embeddings)
            
            if existing_embeddings is not None:
                all_embeddings = np.vstack([existing_embeddings, combined_new])
                all_pmids = existing_pmids + new_pmids
            else:
                all_embeddings = combined_new
                all_pmids = new_pmids
            
            np.save(self.embeddings_path, all_embeddings)
            with open(self.pmid_index_path, 'w') as f:
                json.dump(all_pmids, f)
    
    def get_stats(self) -> dict:
        """Get embedding statistics"""
        stats = {}
        
        if self.embeddings_path.exists():
            embeddings = np.load(self.embeddings_path)
            stats['total_embeddings'] = embeddings.shape[0]
            stats['embedding_dimension'] = embeddings.shape[1]
            stats['file_size_mb'] = self.embeddings_path.stat().st_size / (1024 * 1024)
        else:
            stats['total_embeddings'] = 0
            stats['embedding_dimension'] = 0
            stats['file_size_mb'] = 0
        
        return stats


if __name__ == "__main__":
    generator = EmbeddingGenerator()
    
    print("\n" + "="*60)
    print("EMBEDDING GENERATOR")
    print("="*60)
    
    stats = generator.get_stats()
    print(f"\nCurrent embeddings: {stats['total_embeddings']:,}")
    
    response = input("\nGenerate/resume embeddings? (yes/no): ").strip().lower()
    if response == "yes":
        generator.generate_all_embeddings()
        
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        stats = generator.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("Generation cancelled.")
