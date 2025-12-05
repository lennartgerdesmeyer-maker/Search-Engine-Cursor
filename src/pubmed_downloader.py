"""
PubMed Article Downloader
Downloads articles matching the MeSH search criteria from PubMed
"""
import time
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from Bio import Entrez
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    NCBI_EMAIL, NCBI_API_KEY, MESH_SEARCH_QUERY,
    DATA_DIR, LOG_DIR, PUBMED_BATCH_SIZE, SAVE_CHECKPOINT_EVERY,
    METADATA_DB_PATH, ARTICLES_PARQUET_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'pubmed_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Entrez
Entrez.email = NCBI_EMAIL
Entrez.api_key = NCBI_API_KEY


class PubMedDownloader:
    """Downloads and stores PubMed articles matching search criteria"""
    
    def __init__(self):
        self.db_path = METADATA_DB_PATH
        self.articles_path = ARTICLES_PARQUET_PATH
        self.checkpoint_path = DATA_DIR / "download_checkpoint.json"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                pmid TEXT PRIMARY KEY,
                doi TEXT,
                title TEXT,
                abstract TEXT,
                authors TEXT,
                journal TEXT,
                pub_date TEXT,
                pub_year INTEGER,
                pub_type TEXT,
                mesh_terms TEXT,
                keywords TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pub_year ON articles(pub_year)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def get_total_count(self, query: str = None) -> int:
        """Get total number of articles matching the query"""
        if query is None:
            query = MESH_SEARCH_QUERY
        
        handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
        results = Entrez.read(handle)
        handle.close()
        
        count = int(results["Count"])
        logger.info(f"Total articles matching query: {count:,}")
        return count
    
    def get_all_pmids(self, query: str = None, max_results: Optional[int] = None) -> List[str]:
        """Retrieve all PMIDs matching the search query by breaking into date ranges"""
        if query is None:
            query = MESH_SEARCH_QUERY
        
        total = self.get_total_count(query)
        if max_results:
            total = min(total, max_results)
        
        logger.info(f"Fetching {total:,} PMIDs...")
        logger.info("Breaking search into date ranges to work around PubMed's 9,999 limit...")
        
        all_pmids = set()  # Use set to avoid duplicates
        batch_size = 9000  # Use 9000 to stay safely under 9999 limit
        
        # Search by year ranges (from 1950 to current year)
        current_year = datetime.now().year
        start_year = 1950
        
        for year in tqdm(range(start_year, current_year + 1), desc="Searching by year"):
            year_query = f"{query} AND {year}[PDAT]"
            
            try:
                # Get count for this year
                handle = Entrez.esearch(db="pubmed", term=year_query, retmax=0)
                results = Entrez.read(handle)
                handle.close()
                year_count = int(results["Count"])
                
                if year_count == 0:
                    continue
                
                if year_count <= batch_size:
                    # Can get all in one go
                    handle = Entrez.esearch(db="pubmed", term=year_query, retmax=year_count)
                    results = Entrez.read(handle)
                    handle.close()
                    all_pmids.update(results["IdList"])
                    logger.debug(f"Year {year}: {year_count} articles")
                else:
                    # Need to break into smaller chunks (by month)
                    logger.info(f"Year {year}: {year_count} articles - breaking into months...")
                    for month in range(1, 13):
                        month_query = f"{query} AND {year}/{month:02d}[PDAT]"
                        handle = Entrez.esearch(db="pubmed", term=month_query, retmax=0)
                        results = Entrez.read(handle)
                        handle.close()
                        month_count = int(results["Count"])
                        
                        if month_count == 0:
                            continue
                        
                        if month_count <= batch_size:
                            handle = Entrez.esearch(db="pubmed", term=month_query, retmax=month_count)
                            results = Entrez.read(handle)
                            handle.close()
                            all_pmids.update(results["IdList"])
                        else:
                            # Still too many - break by day ranges
                            logger.warning(f"Year {year}, Month {month}: {month_count} articles - using day ranges...")
                            days_per_chunk = 10
                            for day_start in range(1, 32, days_per_chunk):
                                day_end = min(day_start + days_per_chunk - 1, 31)
                                day_query = f"{query} AND {year}/{month:02d}/{day_start:02d}:{year}/{month:02d}/{day_end:02d}[PDAT]"
                                try:
                                    handle = Entrez.esearch(db="pubmed", term=day_query, retmax=batch_size)
                                    results = Entrez.read(handle)
                                    handle.close()
                                    all_pmids.update(results["IdList"])
                                except:
                                    pass
                        
                        time.sleep(0.1)
                
                time.sleep(0.1)
                
                if max_results and len(all_pmids) >= max_results:
                    all_pmids = set(list(all_pmids)[:max_results])
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching PMIDs for year {year}: {e}")
                time.sleep(1)
                continue
        
        pmids_list = list(all_pmids)
        logger.info(f"Retrieved {len(pmids_list):,} unique PMIDs")
        return pmids_list
    
    def fetch_article_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch detailed metadata for a batch of PMIDs"""
        if not pmids:
            return []
        
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pmids),
                rettype="xml",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()
            
            articles = []
            for record in records.get("PubmedArticle", []):
                article = self._parse_article(record)
                if article:
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching article details: {e}")
            return []
    
    def _parse_article(self, record: Dict) -> Optional[Dict[str, Any]]:
        """Parse a PubMed record into a structured dictionary"""
        try:
            medline = record.get("MedlineCitation", {})
            article_data = medline.get("Article", {})
            
            # PMID
            pmid = str(medline.get("PMID", ""))
            if not pmid:
                return None
            
            # DOI
            doi = ""
            for id_obj in record.get("PubmedData", {}).get("ArticleIdList", []):
                if id_obj.attributes.get("IdType") == "doi":
                    doi = str(id_obj)
                    break
            
            # Title
            title = str(article_data.get("ArticleTitle", ""))
            
            # Abstract
            abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_parts, list):
                abstract = " ".join(str(part) for part in abstract_parts)
            else:
                abstract = str(abstract_parts)
            
            # Authors
            authors_list = article_data.get("AuthorList", [])
            authors = []
            for author in authors_list:
                last = author.get("LastName", "")
                first = author.get("ForeName", "")
                if last:
                    authors.append(f"{last}, {first}" if first else last)
            authors_str = "; ".join(authors)
            
            # Journal
            journal_info = article_data.get("Journal", {})
            journal = journal_info.get("Title", "")
            
            # Publication Date
            pub_date_info = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = pub_date_info.get("Year", "")
            month = pub_date_info.get("Month", "01")
            day = pub_date_info.get("Day", "01")
            
            month_map = {
                "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
            }
            if month in month_map:
                month = month_map[month]
            
            try:
                pub_year = int(year) if year else None
                pub_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}" if year else ""
            except:
                pub_year = None
                pub_date = ""
            
            # Publication Types
            pub_types = article_data.get("PublicationTypeList", [])
            pub_type_str = "; ".join(str(pt) for pt in pub_types)
            
            # MeSH Terms
            mesh_list = medline.get("MeshHeadingList", [])
            mesh_terms = []
            for mesh in mesh_list:
                descriptor = mesh.get("DescriptorName", "")
                if descriptor:
                    mesh_terms.append(str(descriptor))
            mesh_str = "; ".join(mesh_terms)
            
            # Keywords
            keyword_list = medline.get("KeywordList", [])
            keywords = []
            for kw_group in keyword_list:
                for kw in kw_group:
                    keywords.append(str(kw))
            keywords_str = "; ".join(keywords)
            
            return {
                "pmid": pmid,
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "authors": authors_str,
                "journal": journal,
                "pub_date": pub_date,
                "pub_year": pub_year,
                "pub_type": pub_type_str,
                "mesh_terms": mesh_str,
                "keywords": keywords_str
            }
            
        except Exception as e:
            logger.error(f"Error parsing article: {e}")
            return None
    
    def save_articles_to_db(self, articles: List[Dict[str, Any]]):
        """Save articles to SQLite database"""
        if not articles:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for article in articles:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO articles 
                    (pmid, doi, title, abstract, authors, journal, 
                     pub_date, pub_year, pub_type, mesh_terms, keywords)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article["pmid"],
                    article["doi"],
                    article["title"],
                    article["abstract"],
                    article["authors"],
                    article["journal"],
                    article["pub_date"],
                    article["pub_year"],
                    article["pub_type"],
                    article["mesh_terms"],
                    article["keywords"]
                ))
            except Exception as e:
                logger.error(f"Error saving article {article.get('pmid')}: {e}")
        
        conn.commit()
        conn.close()
    
    def save_checkpoint(self, processed_count: int, total_count: int):
        """Save download progress checkpoint"""
        checkpoint = {
            "processed_count": processed_count,
            "total_count": total_count,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
    
    def load_checkpoint(self) -> Optional[int]:
        """Load previous checkpoint if exists"""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Resuming from checkpoint: {checkpoint['processed_count']:,} articles")
            return checkpoint["processed_count"]
        return None
    
    def get_existing_pmids(self) -> set:
        """Get set of PMIDs already in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT pmid FROM articles")
        existing = {row[0] for row in cursor.fetchall()}
        conn.close()
        return existing
    
    def download_all(self, resume: bool = True, max_articles: Optional[int] = None):
        """Main download function"""
        logger.info("=" * 60)
        logger.info("STARTING PUBMED DOWNLOAD")
        logger.info("=" * 60)
        
        all_pmids = self.get_all_pmids(max_results=max_articles)
        
        if not all_pmids:
            logger.error("No PMIDs found. Check your search query.")
            return
        
        if resume:
            existing_pmids = self.get_existing_pmids()
            pmids_to_fetch = [p for p in all_pmids if p not in existing_pmids]
            logger.info(f"Already have {len(existing_pmids):,} articles. Need to fetch {len(pmids_to_fetch):,} more.")
        else:
            pmids_to_fetch = all_pmids
        
        if not pmids_to_fetch:
            logger.info("All articles already downloaded!")
            return
        
        total_batches = (len(pmids_to_fetch) + PUBMED_BATCH_SIZE - 1) // PUBMED_BATCH_SIZE
        processed = 0
        
        logger.info(f"Downloading {len(pmids_to_fetch):,} articles in {total_batches:,} batches...")
        
        for i in tqdm(range(0, len(pmids_to_fetch), PUBMED_BATCH_SIZE), desc="Downloading"):
            batch_pmids = pmids_to_fetch[i:i + PUBMED_BATCH_SIZE]
            
            articles = self.fetch_article_details(batch_pmids)
            self.save_articles_to_db(articles)
            
            processed += len(batch_pmids)
            
            if processed % SAVE_CHECKPOINT_EVERY == 0:
                self.save_checkpoint(processed, len(pmids_to_fetch))
            
            time.sleep(0.15)
        
        logger.info("=" * 60)
        logger.info(f"DOWNLOAD COMPLETE: {processed:,} articles")
        logger.info("=" * 60)
    
    def export_to_parquet(self):
        """Export database to Parquet format"""
        logger.info("Exporting to Parquet format...")
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM articles", conn)
        conn.close()
        
        df.to_parquet(self.articles_path, index=False)
        logger.info(f"Exported {len(df):,} articles to {self.articles_path}")
        
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM articles")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM articles WHERE abstract != ''")
        with_abstract = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(pub_year), MAX(pub_year) FROM articles WHERE pub_year IS NOT NULL")
        year_range = cursor.fetchone()
        
        cursor.execute("""
            SELECT pub_year, COUNT(*) FROM articles 
            WHERE pub_year IS NOT NULL 
            GROUP BY pub_year ORDER BY pub_year DESC LIMIT 10
        """)
        recent_years = cursor.fetchall()
        
        conn.close()
        
        stats = {
            "total_articles": total,
            "with_abstract": with_abstract,
            "without_abstract": total - with_abstract,
            "year_range": f"{year_range[0]} - {year_range[1]}" if year_range[0] else "N/A",
            "recent_years": dict(recent_years)
        }
        
        return stats


def add_custom_articles(pmids: List[str]):
    """Add specific articles by PMID that weren't captured by MeSH search."""
    downloader = PubMedDownloader()
    
    existing = downloader.get_existing_pmids()
    new_pmids = [p for p in pmids if p not in existing]
    
    if not new_pmids:
        logger.info("All specified articles already in database")
        return
    
    logger.info(f"Adding {len(new_pmids)} new articles...")
    
    for i in range(0, len(new_pmids), PUBMED_BATCH_SIZE):
        batch = new_pmids[i:i + PUBMED_BATCH_SIZE]
        articles = downloader.fetch_article_details(batch)
        downloader.save_articles_to_db(articles)
        time.sleep(0.15)
    
    logger.info(f"Added {len(new_pmids)} articles to database")


if __name__ == "__main__":
    downloader = PubMedDownloader()
    
    print("\n" + "="*60)
    print("PUBMED DOWNLOADER FOR FOOT & ANKLE LITERATURE")
    print("="*60)
    
    count = downloader.get_total_count()
    print(f"\nArticles matching search criteria: {count:,}")
    
    response = input("\nStart download? (yes/no): ").strip().lower()
    if response == "yes":
        downloader.download_all()
        downloader.export_to_parquet()
        
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        stats = downloader.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("Download cancelled.")

