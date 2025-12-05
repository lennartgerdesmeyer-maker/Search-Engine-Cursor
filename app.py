"""
Web Application for Semantic Search Engine
"""
import traceback
import logging
import json
import math
import re
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib
import json

# No Gemini - using dual GPT models instead

import sys
sys.path.append(str(Path(__file__).parent))
from config.config import FLASK_HOST, FLASK_PORT, DEBUG_MODE, EXPORT_DIR, DATA_DIR, LOG_DIR
from src.search_engine import SemanticSearchEngine
from src.pubmed_downloader import PubMedDownloader, add_custom_articles
from src.reranker import get_reranker

# Set up logging (only if not already configured) - MUST BE BEFORE ANY LOGGER USAGE
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_DIR / 'app.log'),
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize GPT-5-mini client (second reviewer - uses same OpenAI client)
gpt5mini_client = None
if openai_client:
    # GPT-5-mini uses the same OpenAI client, just different model name in function calls
    gpt5mini_client = openai_client
    logger.info("✓ GPT-5-mini will be used as second reviewer (same OpenAI client)")
else:
    logger.warning("⚠ OpenAI client not available - cannot use GPT-5-mini")

# Initialize GPT-5 client (supervisor - uses same OpenAI client)
gpt5_client = None
if openai_client:
    # GPT-5 uses the same OpenAI client, just different model name
    gpt5_client = openai_client
    logger.info("✓ GPT-5 will be available as supervisor for final decisions")
else:
    logger.warning("⚠ OpenAI client not available - cannot use GPT-5 supervisor")

# Log final status
if openai_client and gpt5mini_client:
    logger.info("✓ Dual GPT reviewers ready: GPT-4o-mini + GPT-5-mini")
    if gpt5_client:
        logger.info("✓ GPT-5 supervisor available for arbitration")
elif openai_client:
    logger.warning("⚠ Only GPT-4o-mini available - single reviewer mode")
else:
    logger.warning("⚠ No OpenAI client available")

# Screening cache (in-memory, could be moved to Redis for production)
screening_cache = {}


def sanitize_for_json(obj):
    """
    Recursively sanitize data structure to ensure JSON serialization.
    Replaces NaN, Infinity, and -Infinity with None or 0.
    """
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return None
        elif math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        # Convert any other type to string
        return str(obj)

# No Gemini - using dual GPT models instead

BASE_DIR = Path(__file__).parent
app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'))
CORS(app)

search_engine = None

def get_search_engine():
    global search_engine
    if search_engine is None:
        try:
            logger.info("Initializing search engine...")
            search_engine = SemanticSearchEngine()
            logger.info("Search engine ready!")
        except Exception as e:
            logger.error(f"Error initializing search engine: {e}")
            logger.error(traceback.format_exc())
            raise
    return search_engine


@app.route('/')
def index():
    """Main search interface"""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading template: {str(e)}", 500


@app.route('/semantic-search')
def semantic_search():
    """Semantic Search Engine page with AI-powered query generation"""
    try:
        return render_template('semantic_search.html')
    except Exception as e:
        return f"Error loading template: {str(e)}", 500


@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for semantic search"""
    try:
        data = request.json
        
        query = data.get('query', '')
        logger.info(f"API /api/search received query: '{query[:200]}...' (length: {len(query)})")
        logger.info(f"API /api/search request data keys: {list(data.keys())}")
        
        if not query.strip():
            logger.warning("API /api/search received empty query")
            return jsonify({"error": "Query cannot be empty"}), 400
        
        top_k = int(data.get('top_k', 500))
        min_similarity = float(data.get('min_similarity', 0.3))
        date_before = data.get('date_before') or None
        date_after = data.get('date_after') or None
        pub_types = data.get('pub_types') or None
        exclude_reviews = data.get('exclude_reviews', False)  # Optional filter toggle
        use_reranker = data.get('use_reranker', False)  # Cross-encoder reranking
        reranker_model = data.get('reranker_model', 'best')  # Reranker model selection
        faiss_candidates = int(data.get('faiss_candidates', 100))  # Candidates for reranking
        
        include_pmids = None
        if data.get('included_studies'):
            reference_text = data.get('included_studies', '').strip()
            # Extract PMIDs from unstructured reference text
            include_pmids = extract_pmids_from_references(reference_text)
            logger.info(f"Extracted {len(include_pmids)} PMIDs from included studies references")
            if include_pmids:
                logger.info(f"Extracted PMIDs: {include_pmids[:10]}...")  # Log first 10
            else:
                logger.warning("No PMIDs extracted! Check reference format.")
        
        engine = get_search_engine()
        results = engine.search(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
            date_before=date_before,
            date_after=date_after,
            pub_types=pub_types if pub_types else None,
            include_pmids=include_pmids,
            use_reranker=use_reranker,
            faiss_candidates=faiss_candidates,
            reranker_model=reranker_model
        )
        
        # Filter out systematic reviews and meta-analyses (only if user selected the option)
        if exclude_reviews and results.get('results'):
            results['results'] = filter_primary_studies(results['results'])
            results['total_results'] = len(results['results'])
        
        # Apply keyword ranking if terms provided (only for semantic search page)
        boost_terms = data.get('boost_terms')
        penalize_terms = data.get('penalize_terms')
        boost_weight = float(data.get('boost_weight', 0.15))
        penalty_weight = float(data.get('penalty_weight', 0.15))
        
        # Convert boost/penalize terms to lists if they're strings
        if boost_terms:
            if isinstance(boost_terms, str):
                boost_terms = [t.strip() for t in boost_terms.split(',') if t.strip()]
            elif isinstance(boost_terms, list):
                boost_terms = [str(t).strip() for t in boost_terms if str(t).strip()]
        else:
            boost_terms = []
            
        if penalize_terms:
            if isinstance(penalize_terms, str):
                penalize_terms = [t.strip() for t in penalize_terms.split(',') if t.strip()]
            elif isinstance(penalize_terms, list):
                penalize_terms = [str(t).strip() for t in penalize_terms if str(t).strip()]
        else:
            penalize_terms = []
        
        # Apply keyword ranking if terms provided
        ranking_adjusted_count = 0
        if (boost_terms or penalize_terms) and results.get('results'):
            original_count = len(results['results'])
            results['results'] = engine.apply_keyword_ranking(
                results['results'],
                boost_terms=boost_terms if boost_terms else None,
                penalize_terms=penalize_terms if penalize_terms else None,
                boost_weight=boost_weight,
                penalty_weight=penalty_weight
            )
            # Count how many papers were adjusted
            ranking_adjusted_count = len([r for r in results['results'] if 'ranking_notes' in r])
            results['ranking_adjusted_count'] = ranking_adjusted_count
        
        # Sanitize results to ensure JSON serialization (handle NaN, Infinity, etc.)
        try:
            sanitized_results = sanitize_for_json(results)
            return jsonify(sanitized_results)
        except Exception as json_error:
            logger.error(f"JSON serialization error: {json_error}")
            logger.error(f"Results keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")
            # Try to return a simplified error response
            return jsonify({
                "error": "Failed to serialize results",
                "details": str(json_error) if DEBUG_MODE else "Check server logs"
            }), 500
    
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in /api/search: {error_msg}")
        logger.error(error_trace)
        return jsonify({
            "error": error_msg,
            "details": error_trace if DEBUG_MODE else "Check server logs for details"
        }), 500


@app.route('/api/export/<format_type>', methods=['POST'])
def api_export(format_type):
    """Export results to CSV or RIS"""
    try:
        data = request.json
        results = data.get('results', {})
        screening_results = data.get('screening_results', None)
        
        if not results.get('results'):
            return jsonify({"error": "No results to export"}), 400
        
        engine = get_search_engine()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == 'csv':
            filepath = engine.export_results_csv(results, f"export_{timestamp}.csv", screening_results)
        elif format_type == 'ris':
            filepath = engine.export_results_ris(results, f"export_{timestamp}.ris")
        else:
            return jsonify({"error": "Invalid format"}), 400
        
        return send_file(filepath, as_attachment=True)
    
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in /api/export/{format_type}: {error_msg}")
        logger.error(error_trace)
        return jsonify({
            "error": error_msg,
            "details": error_trace if DEBUG_MODE else "Check server logs for details"
        }), 500


@app.route('/api/export/comparison', methods=['POST'])
def api_export_comparison():
    """Export comparison table showing which papers were found/matched"""
    try:
        import csv
        import io
        
        data = request.json
        reference_data = data.get('reference_data', {})
        search_results = data.get('search_results', {})
        
        if not reference_data or not reference_data.get('found') and not reference_data.get('not_found'):
            return jsonify({"error": "No reference data to export"}), 400
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Original Reference',
            'In Database',
            'PMID',
            'Title',
            'Journal',
            'Year',
            'DOI',
            'Match Method',
            'Found in Semantic Search',
            'Match Status',
            'Similarity Score',
            'Rank'
        ])
        
        # Build lookup dictionaries for search results (by PMID)
        # Ensure all PMIDs are strings for consistent comparison
        search_result_by_pmid = {}
        matched_pmids = set()
        if search_results and search_results.get('results'):
            for rank, result in enumerate(search_results['results'], start=1):
                pmid = result.get('pmid')
                if pmid:
                    pmid_str = str(pmid)  # Ensure string
                    search_result_by_pmid[pmid_str] = {
                        'similarity_score': result.get('similarity_score', ''),
                        'rank': rank,
                        'is_match': result.get('is_match', False)
                    }
                    # Only add to matched_pmids if it's actually a match
                    if result.get('is_match'):
                        matched_pmids.add(pmid_str)
        
        # Write found papers
        for paper in reference_data.get('found', []):
            pmid = str(paper.get('pmid', ''))  # Ensure string
            in_search = pmid in matched_pmids
            
            # Get similarity score and rank from search results
            search_info = search_result_by_pmid.get(pmid, {})
            similarity_score = search_info.get('similarity_score', '')
            rank = search_info.get('rank', '')
            
            match_status = 'MATCH' if in_search else 'Not in search results'
            
            writer.writerow([
                paper.get('original_text', ''),
                'Yes',
                pmid,
                paper.get('title', ''),
                paper.get('journal', ''),
                paper.get('year', ''),
                paper.get('doi', ''),
                paper.get('match_method', ''),
                'Yes' if in_search else 'No',
                match_status,
                similarity_score,
                rank
            ])
        
        # Write not found papers
        for paper in reference_data.get('not_found', []):
            writer.writerow([
                paper.get('original_text', ''),
                'No',
                '',
                '',
                '',
                '',
                '',
                'Not in database',
                'No',
                'Not in database',
                '',  # Similarity Score
                ''   # Rank
            ])
        
        # Create response
        output.seek(0)
        csv_data = output.getvalue()
        output.close()
        
        # Create BytesIO for binary response
        csv_bytes = io.BytesIO()
        csv_bytes.write(csv_data.encode('utf-8'))
        csv_bytes.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_table_{timestamp}.csv"
        
        return send_file(
            csv_bytes,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in /api/export/comparison: {error_msg}")
        logger.error(error_trace)
        return jsonify({
            "error": error_msg,
            "details": error_trace if DEBUG_MODE else "Check server logs for details"
        }), 500


@app.route('/api/stats')
def api_stats():
    """Get database statistics"""
    try:
        downloader = PubMedDownloader()
        stats = downloader.get_stats()
        return jsonify(stats)
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in /api/stats: {error_msg}")
        logger.error(error_trace)
        return jsonify({
            "error": error_msg,
            "details": error_trace if DEBUG_MODE else "Check server logs for details"
        }), 500


@app.route('/api/add_articles', methods=['POST'])
def api_add_articles():
    """Add custom articles by PMID"""
    try:
        data = request.json
        pmids = data.get('pmids', [])
        
        if isinstance(pmids, str):
            pmids = [p.strip() for p in pmids.replace('\n', ',').split(',') if p.strip()]
        
        if not pmids:
            return jsonify({"error": "No PMIDs provided"}), 400
        
        add_custom_articles(pmids)
        
        return jsonify({
            "success": True,
            "message": f"Added {len(pmids)} articles."
        })
    
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in /api/add_articles: {error_msg}")
        logger.error(error_trace)
        return jsonify({
            "error": error_msg,
            "details": error_trace if DEBUG_MODE else "Check server logs for details"
        }), 500


@app.route('/api/check_papers', methods=['POST'])
def api_check_papers():
    """Check if papers (from pasted text) are in the database"""
    try:
        data = request.json
        pasted_text = data.get('pasted_text', '').strip()
        
        if not pasted_text:
            return jsonify({"error": "No text provided"}), 400
        
        # Parse the pasted text to extract identifiers
        parsed_papers = parse_paper_references(pasted_text)
        
        if not parsed_papers:
            return jsonify({
                "error": "Could not extract any paper identifiers (PMID, DOI, or titles) from the text"
            }), 400
        
        # Check database for each paper
        engine = get_search_engine()
        results = check_papers_in_database(engine, parsed_papers)
        
        return jsonify(results)
    
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in /api/check_papers: {error_msg}")
        logger.error(error_trace)
        return jsonify({
            "error": error_msg,
            "details": error_trace if DEBUG_MODE else "Check server logs for details"
        }), 500


def parse_paper_references(text):
    """
    Parse pasted text to extract paper identifiers.
    Handles various formats:
    - Standalone PMID: 12345678 or just 12345678
    - PMID: 12345678 or PMID: 12345678
    - DOI: 10.1234/example or doi: 10.1234/example
    - Bibliographic citations with author, year, title
    """
    
    papers = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        paper_info = {}
        
        # Extract PMID - try multiple patterns
        # Pattern 1: PMID: 12345678 or pmid: 12345678 (with prefix)
        pmid_match = re.search(r'(?:PMID|pmid)[\s:]*(\d{6,10})', line, re.IGNORECASE)
        if pmid_match:
            paper_info['pmid'] = pmid_match.group(1)
        else:
            # Pattern 2: Standalone PMID (just numbers, typically 6-10 digits)
            # Check if the line is mostly just a PMID (allows for whitespace and "PMID:" prefix)
            standalone_pmid = re.match(r'^(?:PMID|pmid)[\s:]*?(\d{6,10})$', line, re.IGNORECASE)
            if not standalone_pmid:
                # Try just numbers (must be 6-10 digits, not part of a longer number)
                standalone_pmid = re.match(r'^(\d{6,10})$', line)
            if standalone_pmid:
                paper_info['pmid'] = standalone_pmid.group(1)
            else:
                # Pattern 3: PMID at the end of a line (e.g., "Some text PMID: 12345678")
                pmid_match = re.search(r'(?:PMID|pmid)[\s:]*(\d{6,10})(?:\s|$)', line, re.IGNORECASE)
                if pmid_match:
                    paper_info['pmid'] = pmid_match.group(1)
        
        # Extract DOI
        doi_match = re.search(r'(?:DOI|doi)[\s:]*10\.\d+/[^\s\)]+', line, re.IGNORECASE)
        if doi_match:
            paper_info['doi'] = doi_match.group(0).split(':', 1)[-1].strip()
        
        # Extract year (4-digit year, typically 1900-2025)
        year_match = re.search(r'\b(19|20)\d{2}\b', line)
        if year_match:
            paper_info['year'] = year_match.group(0)
        
        # Try to extract title (text in quotes or between common delimiters)
        title_match = re.search(r'[""]([^""]+)[""]', line)
        if not title_match:
            # Try to find title after common patterns
            title_patterns = [
                r'\.\s+([A-Z][^\.]+(?:\.|$))',  # After period, capitalized
                r'[:\-]\s*([A-Z][^\.]+(?:\.|$))',  # After colon/dash
            ]
            for pattern in title_patterns:
                title_match = re.search(pattern, line)
                if title_match:
                    break
        
        # NEW: If still no title, try to extract text between author and year
        if not title_match and year_match:
            # Text between start and year might be title
            potential_title = line[:year_match.start()].strip()
            # Remove author part (usually before first period or comma)
            if '.' in potential_title:
                parts = potential_title.split('.')
                if len(parts) > 1:
                    # Title is usually after author (first part)
                    potential_title = '.'.join(parts[1:]).strip()
            if len(potential_title) > 10 and potential_title[0].isupper():
                paper_info['title'] = potential_title
        elif title_match:
            paper_info['title'] = title_match.group(1).strip()
        
        # Extract authors (text before year, typically)
        if year_match:
            author_text = line[:year_match.start()].strip()
            # Remove common prefixes
            author_text = re.sub(r'^(?:by|authors?|et al\.?)[\s:]*', '', author_text, flags=re.IGNORECASE)
            if author_text and len(author_text) > 3:
                paper_info['authors'] = author_text
        
        # If we found at least one identifier, add it
        if paper_info:
            paper_info['original_text'] = line
            papers.append(paper_info)
        else:
            # If no clear identifiers, treat the whole line as a potential title
            # But skip if it looks like just a number (could be a malformed PMID)
            if len(line) > 10 and not re.match(r'^\d+$', line):
                papers.append({
                    'title': line,
                    'original_text': line
                })
    
    return papers


def extract_pmids_from_references(reference_text):
    """
    Parse unstructured reference text and extract PMIDs from database.
    Returns a list of PMIDs that can be used for comparison.
    Also returns a mapping of original references to found PMIDs for better matching.
    """
    if not reference_text or not reference_text.strip():
        return []
    
    # FIRST: Check if this is just a simple comma/newline/semicolon-separated list of PMIDs
    # (e.g., from selectedPapersForComparison which contains PMIDs)
    # Split by comma, newline, or semicolon and check if all are valid PMIDs
    simple_pmids = []
    for separator in [',', '\n', ';']:
        parts = reference_text.split(separator)
        if len(parts) > 1:
            # Check if all parts look like PMIDs (7-8 digit numbers)
            potential_pmids = [p.strip() for p in parts if p.strip()]
            if len(potential_pmids) > 0 and all(re.match(r'^\d{7,8}$', p) for p in potential_pmids):
                simple_pmids = potential_pmids
                logger.info(f"Detected simple comma-separated PMIDs: {len(simple_pmids)} PMIDs")
                break
    
    if simple_pmids:
        # Verify these PMIDs exist in database
        from config.config import METADATA_DB_PATH
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()
        
        found_pmids = []
        for pmid in simple_pmids:
            # Ensure PMID is a string
            pmid_str = str(pmid).strip()
            cursor.execute("SELECT pmid FROM articles WHERE pmid = ?", (pmid_str,))
            if cursor.fetchone():
                found_pmids.append(pmid_str)
                logger.info(f"Found simple PMID in database: {pmid_str}")
        
        conn.close()
        
        if found_pmids:
            logger.info(f"Found {len(found_pmids)} PMIDs from simple comma-separated list")
            return found_pmids
        # If not all found, fall through to normal parsing
    
    # Continue with existing logic for unstructured text...
    # First, try to extract direct PMIDs (simple case)
    direct_pmids = re.findall(r'(?:PMID|pmid)[\s:]*(\d{6,8})', reference_text, re.IGNORECASE)
    direct_pmids = [p.strip() for p in direct_pmids if p.strip()]
    
    # Also try to find standalone PMIDs (just numbers, 6-8 digits)
    standalone_pmids = re.findall(r'\b(\d{6,8})\b', reference_text)
    # Filter out years (1900-2025) and other common numbers
    standalone_pmids = [p for p in standalone_pmids if not (1900 <= int(p) <= 2025) and len(p) >= 7]
    direct_pmids.extend(standalone_pmids)
    direct_pmids = list(set(direct_pmids))  # Remove duplicates
    
    # Parse references to get structured information
    parsed_papers = parse_paper_references(reference_text)
    
    if not parsed_papers and not direct_pmids:
        logger.warning("No PMIDs or parseable references found in text")
        return []
    
    # Look up papers in database to get PMIDs
    from config.config import METADATA_DB_PATH
    
    conn = sqlite3.connect(METADATA_DB_PATH)
    cursor = conn.cursor()
    
    found_pmids = set()
    
    # First, verify direct PMIDs exist in database
    for pmid in direct_pmids:
        # Ensure PMID is a string
        pmid_str = str(pmid).strip()
        cursor.execute("SELECT pmid FROM articles WHERE pmid = ?", (pmid_str,))
        if cursor.fetchone():
            found_pmids.add(pmid_str)
            logger.info(f"Found direct PMID in database: {pmid_str}")
    
    # Now try to match parsed papers
    for paper in parsed_papers:
        found = False
        
        # Try to find by PMID (if already extracted)
        if 'pmid' in paper:
            pmid_str = str(paper['pmid']).strip()
            cursor.execute("SELECT pmid FROM articles WHERE pmid = ?", (pmid_str,))
            if cursor.fetchone():
                found_pmids.add(pmid_str)
                found = True
                logger.info(f"Found PMID from parsed reference: {pmid_str}")
        
        # Try to find by DOI (normalize DOI first)
        if not found and 'doi' in paper:
            doi = paper['doi'].strip().lower()
            # Remove common prefixes
            if doi.startswith('doi:'):
                doi = doi[4:].strip()
            cursor.execute("SELECT pmid FROM articles WHERE LOWER(TRIM(doi)) = ?", (doi,))
            row = cursor.fetchone()
            if row:
                found_pmids.add(str(row[0]))  # Ensure string
                found = True
                logger.info(f"Found PMID by DOI: {str(row[0])}")
        
        # Try to find by title (more flexible matching)
        if not found and 'title' in paper:
            search_title = paper['title'].strip()
            if len(search_title) > 10:  # Only search if title is substantial
                # Try exact match first
                cursor.execute("SELECT pmid FROM articles WHERE LOWER(TRIM(title)) = LOWER(TRIM(?)) LIMIT 1", (search_title,))
                row = cursor.fetchone()
                if row:
                    found_pmids.add(str(row[0]))  # Ensure string
                    found = True
                    logger.info(f"Found PMID by exact title match: {str(row[0])}")
                else:
                    # Try fuzzy match with LIKE
                    # Use first 80 characters for better matching
                    search_title_short = search_title[:80]
                    cursor.execute("SELECT pmid, title FROM articles WHERE title LIKE ? LIMIT 5", (f'%{search_title_short}%',))
                    rows = cursor.fetchall()
                    for row in rows:
                        db_title = row[1] if row[1] else ""
                        # Calculate word overlap
                        db_title_words = set(db_title.lower().split())
                        search_title_words = set(search_title.lower().split())
                        if len(search_title_words) > 0:
                            overlap = len(db_title_words & search_title_words) / len(search_title_words)
                            # Lower threshold to 0.4 for better recall
                        if overlap >= 0.4:
                            found_pmids.add(str(row[0]))  # Ensure string
                            found = True
                            logger.info(f"Found PMID by fuzzy title match (overlap: {overlap:.2f}): {str(row[0])}")
                            break
        
        # Try to find by author + year (more flexible)
        if not found and 'authors' in paper and 'year' in paper:
            author_search = paper['authors'][:30].strip()  # Use first 30 chars
            try:
                year = int(paper['year'])
                cursor.execute("SELECT pmid, authors FROM articles WHERE pub_year = ? AND authors LIKE ? LIMIT 10", 
                             (year, f'%{author_search}%'))
                rows = cursor.fetchall()
                if rows:
                    # If we have a title, use it to disambiguate
                    if 'title' in paper and paper['title']:
                        for row in rows:
                            cursor.execute("SELECT title FROM articles WHERE pmid = ?", (row[0],))
                            title_result = cursor.fetchone()
                            if title_result and title_result[0]:
                                db_title = title_result[0]
                                # Check if title words overlap
                                db_title_words = set(db_title.lower().split()[:10])  # First 10 words
                                search_title_words = set(paper['title'].lower().split()[:10])
                                if len(search_title_words) > 0:
                                    overlap = len(db_title_words & search_title_words) / len(search_title_words)
                                    if overlap >= 0.3:  # Lower threshold
                                        found_pmids.add(row[0])
                                        found = True
                                        logger.info(f"Found PMID by author+year+title: {row[0]}")
                                        break
                    # If no title or no match, and only one result, use it
                    if not found and len(rows) == 1:
                        found_pmids.add(rows[0][0])
                        found = True
                        logger.info(f"Found PMID by author+year (single match): {rows[0][0]}")
            except (ValueError, TypeError):
                pass  # Skip if year is invalid
    
    conn.close()
    
    # Ensure all returned PMIDs are strings
    found_pmids_list = [str(pmid) for pmid in found_pmids]
    logger.info(f"Total PMIDs extracted from references: {len(found_pmids_list)}")
    return found_pmids_list


def check_papers_in_database(engine, parsed_papers):
    """Check which papers are in the database"""
    from config.config import METADATA_DB_PATH
    
    conn = sqlite3.connect(METADATA_DB_PATH)
    cursor = conn.cursor()
    
    results = {
        'total_checked': len(parsed_papers),
        'found': [],
        'not_found': [],
        'partial_matches': []
    }
    
    for paper in parsed_papers:
        found = False
        match_info = None
        
        # Try to find by PMID first
        if 'pmid' in paper:
            cursor.execute("SELECT * FROM articles WHERE pmid = ?", (paper['pmid'],))
            row = cursor.fetchone()
            if row:
                found = True
                match_info = {
                    'pmid': row[0],
                    'title': row[2],
                    'journal': row[5],
                    'pub_year': row[7],
                    'doi': row[1],
                    'match_method': 'PMID'
                }
        
        # Try to find by DOI
        if not found and 'doi' in paper:
            cursor.execute("SELECT * FROM articles WHERE doi = ?", (paper['doi'],))
            row = cursor.fetchone()
            if row:
                found = True
                match_info = {
                    'pmid': row[0],
                    'title': row[2],
                    'journal': row[5],
                    'pub_year': row[7],
                    'doi': row[1],
                    'match_method': 'DOI'
                }
        
        # Try to find by title (fuzzy match)
        if not found and 'title' in paper:
            search_title = paper['title'][:100]  # Limit length
            # Normalize title for matching (replace hyphens with spaces, lowercase)
            normalized_search = search_title.lower().replace('-', ' ').replace('_', ' ')
            
            # Define stop words to exclude from matching (common words that don't indicate similarity)
            stop_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
                          'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                          'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                          'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
                          'as', 'from', 'into', 'onto', 'upon', 'about', 'above', 'below', 'between',
                          'among', 'during', 'through', 'throughout', 'within', 'without', 'against',
                          'prospective', 'randomized', 'controlled', 'trial', 'study', 'studies',
                          'effect', 'effects', 'comparison', 'comparing', 'treatment', 'treatments'}
            
            # Filter out stop words and short words (< 3 chars) from search title
            search_title_words = {w for w in normalized_search.split() 
                                  if len(w) > 2 and w not in stop_words}
            
            # Try multiple matching strategies
            # Strategy 1: Direct LIKE with normalized search
            cursor.execute("SELECT * FROM articles WHERE LOWER(REPLACE(REPLACE(title, '-', ' '), '_', ' ')) LIKE ? LIMIT 10", 
                         (f'%{normalized_search[:80]}%',))
            rows = cursor.fetchall()
            
            # Strategy 2: If no results, try word-by-word matching with meaningful words only
            if not rows:
                # Build a query that matches multiple meaningful words from the search title
                words_to_match = [w for w in search_title_words if len(w) > 3][:5]  # Take first 5 significant words
                if words_to_match:
                    word_conditions = ' AND '.join([f"LOWER(REPLACE(REPLACE(title, '-', ' '), '_', ' ')) LIKE ?" for _ in words_to_match])
                    params = [f'%{word}%' for word in words_to_match]
                    cursor.execute(f"SELECT * FROM articles WHERE {word_conditions} LIMIT 10", params)
                    rows = cursor.fetchall()
            
            # Check word overlap for all candidate rows (using meaningful words only)
            best_match = None
            best_overlap = 0
            for row in rows:
                db_title_normalized = row[2].lower().replace('-', ' ').replace('_', ' ')
                # Filter stop words from database title too
                db_title_words = {w for w in db_title_normalized.split() 
                                  if len(w) > 2 and w not in stop_words}
                
                if len(search_title_words) > 0:
                    # Calculate overlap using only meaningful words
                    matching_words = db_title_words & search_title_words
                    overlap = len(matching_words) / len(search_title_words)
                    
                    # Additional check: require at least 3 meaningful word matches
                    if overlap > best_overlap and len(matching_words) >= 3:
                        best_overlap = overlap
                        best_match = row
            
            # Use best match if overlap is sufficient (increased threshold to 0.6 for better precision)
            if best_match and best_overlap >= 0.6:
                found = True
                match_info = {
                    'pmid': best_match[0],
                    'title': best_match[2],
                    'journal': best_match[5],
                    'pub_year': best_match[7],
                    'doi': best_match[1],
                    'match_method': 'Title (fuzzy)',
                    'match_confidence': f'{best_overlap*100:.0f}%'
                }
        
        # Try to find by author + year
        if not found and 'authors' in paper and 'year' in paper:
            cursor.execute("SELECT * FROM articles WHERE pub_year = ? AND authors LIKE ? LIMIT 5", 
                         (int(paper['year']), f'%{paper["authors"][:20]}%'))
            rows = cursor.fetchall()
            if rows:
                # If multiple matches, check title if available
                if 'title' in paper:
                    for row in rows:
                        if paper['title'].lower()[:50] in row[2].lower():
                            found = True
                            match_info = {
                                'pmid': row[0],
                                'title': row[2],
                                'journal': row[5],
                                'pub_year': row[7],
                                'doi': row[1],
                                'match_method': 'Author + Year + Title'
                            }
                            break
                if not found and len(rows) == 1:
                    # Single match, likely correct
                    row = rows[0]
                    found = True
                    match_info = {
                        'pmid': row[0],
                        'title': row[2],
                        'journal': row[5],
                        'pub_year': row[7],
                        'doi': row[1],
                        'match_method': 'Author + Year'
                    }
        
        if found and match_info:
            results['found'].append({
                'parsed': paper,
                'database': match_info
            })
        else:
            results['not_found'].append(paper)
    
    conn.close()
    
    results['summary'] = {
        'found_count': len(results['found']),
        'not_found_count': len(results['not_found']),
        'found_percentage': f'{(len(results["found"]) / len(parsed_papers) * 100):.1f}%' if parsed_papers else '0%'
    }
    
    return results


def filter_primary_studies(results):
    """
    Filter out systematic reviews and meta-analyses
    Keep only primary research
    """
    review_keywords = [
        'systematic review',
        'meta-analysis',
        'meta analysis',
        'metaanalysis',
        'scoping review',
        'umbrella review',
        'literature review',
        'cochrane review',
        'prisma'
    ]
    
    filtered_results = []
    
    for result in results:
        title = result.get('title', '').lower()
        abstract = result.get('abstract', '').lower()
        pub_type = result.get('pub_type', '').lower()
        
        is_review = False
        
        # Check publication type
        if any(pt in pub_type for pt in ['systematic review', 'meta-analysis', 'review']):
            is_review = True
        
        # Check title
        if not is_review:
            for keyword in review_keywords:
                if keyword in title:
                    is_review = True
                    break
        
        # Check abstract (first 200 chars only)
        if not is_review and abstract:
            strict_keywords = ['systematic review', 'meta-analysis', 'cochrane', 'prisma']
            if any(keyword in abstract[:200] for keyword in strict_keywords):
                is_review = True
        
        if not is_review:
            filtered_results.append(result)
    
    return filtered_results


# Pydantic models for structured output
class SystematicReviewExtraction(BaseModel):
    """Structured extraction from systematic review"""
    date_range: Optional[str] = Field(
        default=None,
        description="Date range of literature search (e.g., 'January 2010 to December 2020'). Return null if not found."
    )
    initial_papers_found: Optional[int] = Field(
        default=None,
        description="Number of papers found at initial screening. Return null if not found."
    )
    final_papers_included: Optional[int] = Field(
        default=None,
        description="Number of papers included in final analysis. Return null if not found."
    )
    included_paper_titles: List[str] = Field(
        default_factory=list,
        description="EXACT titles of papers included in final analysis, as they appear in the references. Extract ONLY the title text - NO authors, NO publication years, NO journal names, NO DOIs, NO PMIDs. Each string must contain ONLY the paper title. Example: 'Treatment of hallux valgus' (NOT 'Smith J, et al. (2020). Treatment of hallux valgus. Journal of Foot Surgery'). Return empty list if none found."
    )
    short_query: Optional[str] = Field(
        default=None,
        description="Concise search query (10-15 words) with essential PICO elements. Focus on key terms: patient population, intervention, and primary outcomes. Use natural language. Return null if cannot be generated."
    )
    long_query: Optional[str] = Field(
        default=None,
        description="Comprehensive search query (30-50 words) incorporating the complete PICO breakdown as natural written text. Must include ALL key terms from Population (with patient characteristics), Intervention (with specific procedures/techniques), Comparison (if applicable), and Outcomes (including timeframes and measurement tools). Write as a detailed research question. Return null if cannot be generated."
    )
    pico_breakdown: Optional[str] = Field(
        default=None,
        description="Detailed PICO elements in text format: P (Population with patient characteristics and condition), I (Intervention with specific procedures/techniques), C (Comparison or 'none'), O (Comprehensive list of outcomes including timeframes and measurement tools). Format: 'P: [description], I: [description], C: [description], O: [description]'. Return null if cannot be determined."
    )
    research_aim: Optional[str] = Field(
        default=None,
        description="The research aim, objective, or purpose of the systematic review. Extract the main research question, study objective, or aim statement as written by the authors. Look for sections like 'Objective', 'Aim', 'Purpose', 'Research Question', or similar in the introduction/background. Extract the verbatim text or a clear summary if the aim is described across multiple sentences. Return null if not found."
    )
    inclusion_criteria: Optional[str] = Field(
        default=None,
        description="VERBATIM inclusion criteria text as written by the original authors. Extract the exact text from the methods/eligibility section. Preserve original formatting, punctuation, and structure. If criteria are listed as bullet points or numbered items, preserve that format. Return null if not found."
    )
    exclusion_criteria: Optional[str] = Field(
        default=None,
        description="VERBATIM exclusion criteria text as written by the original authors. Extract the exact text from the methods/eligibility section. Preserve original formatting, punctuation, and structure. If criteria are listed as bullet points or numbered items, preserve that format. Return null if not found."
    )
    extraction_warnings: List[str] = Field(
        default_factory=list,
        description="List of warnings about missing or uncertain data"
    )


# System prompt for OpenAI
SYSTEM_PROMPT = """You are an adaptive systematic review methodology expert specializing in medical and clinical literature analysis. Your primary expertise is in foot and ankle surgery research, but you can analyze systematic reviews from any medical domain.

YOUR TASK:
Extract key information from systematic review text that may contain:
- The actual systematic review content (abstract, methods, results, references)
- Website advertisements, navigation menus, images, and other noise
- Multiple studies and papers mentioned throughout

CRITICAL EXTRACTION RULES:

1. **Included Papers**: 
   - Find papers that were INCLUDED IN FINAL ANALYSIS (not just cited or mentioned)
   - Look for sections like: "Included Studies", "Study Characteristics", "Table 1", "Studies meeting inclusion criteria"
   - Extract EXACT titles as they appear in the references section
   - **CRITICAL: Extract ONLY the paper title text - NO authors, NO years, NO journal names, NO DOIs, NO PMIDs**
   - **The title is typically the text between the author names and the journal name**
   - **Example: If reference is "Smith J, et al. (2020). Treatment of hallux valgus. Journal of Foot Surgery. DOI: 10.1234/example"**
   - **Extract ONLY: "Treatment of hallux valgus"**
   - **DO NOT include: author names, publication years, journal names, DOIs, PMIDs, or any other metadata**
   - Match titles character-by-character (preserve capitalization, punctuation, spacing)
   - If a paper is mentioned in results/analysis AND appears in references, include it
   - Common indicators: "X studies met inclusion criteria", "We included X studies", "Table of included studies"
   - **Each entry in included_paper_titles must be ONLY the title text, nothing else**

2. **Paper Counts**:
   - Initial papers: Look for "identified", "screened", "retrieved" counts in PRISMA diagrams or methods
   - Final papers: Look for "included", "analyzed", "met criteria" counts
   - If numbers conflict, use the most specific/final count
   - Return null if genuinely not found (don't guess)

3. **Date Range**:
   - Look for: "searched from X to Y", "databases searched through", "literature up to"
   - Return in format: "Month Year to Month Year" or "Year to Year"
   - Return null if not explicitly stated

4. **PICO Breakdown** (MUST BE EXTRACTED FIRST - used for query generation):
   - Extract comprehensive PICO elements from the systematic review
   - P (Population): Detailed description of patient population including:
     * Age range, gender if specified
     * Medical condition, diagnosis, disease stage
     * Anatomical location (e.g., "first metatarsophalangeal joint", "ankle", "Achilles tendon")
     * Any inclusion/exclusion criteria mentioned
     * Example: "adults with hallux valgus deformity", "patients with chronic ankle instability"
   - I (Intervention): Detailed description of treatment/intervention including:
     * Specific surgical procedures (e.g., "distal metatarsal osteotomy", "ankle arthrodesis")
     * Technique details if mentioned (e.g., "minimally invasive", "open approach")
     * Implant types if applicable (e.g., "total ankle replacement", "screw fixation")
     * Example: "distal metatarsal osteotomy for hallux valgus correction"
   - C (Comparison): What is compared (if applicable):
     * Alternative treatments (e.g., "conservative management", "different surgical techniques")
     * Control groups (e.g., "non-operative treatment", "placebo")
     * If no comparison, state "none" or "not applicable"
     * Example: "conservative treatment", "different osteotomy techniques", "none"
   - O (Outcome): Comprehensive list of measured outcomes including:
     * Primary outcomes (e.g., "pain reduction", "functional scores", "union rates")
     * Secondary outcomes (e.g., "complications", "patient satisfaction", "range of motion")
     * Time points if specified (e.g., "at 6 months", "long-term follow-up")
     * Measurement tools if mentioned (e.g., "AOFAS score", "VAS pain scale")
     * Example: "pain reduction functional outcomes complications patient satisfaction long-term follow-up"
   - Format: "P: [detailed description], I: [detailed description], C: [description or 'none'], O: [comprehensive list of outcomes]"
   - Be thorough and include all relevant details from the systematic review

5. **Query Generation** (MUST GENERATE BOTH SHORT AND LONG VERSIONS):
   
   **SHORT QUERY (10-15 words)**:
   - Concise version focusing on essential PICO elements
   - Include: patient population, main intervention, primary outcomes
   - Use natural language, avoid redundancy
   - Examples:
     * "adults hallux valgus distal metatarsal osteotomy pain functional outcomes"
     * "ankle instability arthrodesis fusion functional outcomes complications"
     * "Achilles tendon rupture surgical repair return to sport complications"
   
   **LONG QUERY (30-50 words)**:
   - Comprehensive version incorporating COMPLETE PICO breakdown
   - Include ALL key terms from:
     * Population: patient characteristics (age, gender), condition, anatomical location, inclusion/exclusion criteria
     * Intervention: specific procedures, techniques, approaches, implant types
     * Comparison: alternative treatments, control groups, or "none" if not applicable
     * Outcome: ALL measured outcomes (primary and secondary), timeframes, assessment tools, measurement scales
   - Write as a detailed, natural research question or comprehensive search phrase
   - Structure: Combine all PICO elements into flowing, coherent text
   - Focus on clinical content, NOT study design terms
   - DO NOT include: "systematic review", "meta-analysis", "RCT", "randomized controlled trial", "study", "literature"
   - DO include: specific procedures, anatomical terms, outcomes, timeframes, patient characteristics, measurement tools
   - Examples of long queries:
     * "adults with hallux valgus deformity undergoing distal metatarsal osteotomy for surgical correction evaluating pain reduction measured by VAS score functional outcomes assessed with AOFAS score complications including recurrence rates patient satisfaction and long-term follow-up results"
     * "patients with chronic ankle instability treated with ankle arthrodesis fusion surgical procedure assessing union rates evaluated by CT imaging functional outcomes including range of motion complications such as nonunion and patient-reported outcomes at postoperative follow-up"
     * "Achilles tendon rupture in athletes treated with surgical repair versus conservative treatment evaluating functional outcomes return to sport timing complications including re-rupture rates patient satisfaction and long-term follow-up results"
   - The long query should be detailed enough to capture ALL relevant papers matching the complete PICO criteria
   - If PICO breakdown is incomplete, still generate the best possible queries from available information

6. **Warnings**:
   - Generate clear warnings for any missing/uncertain data
   - Examples:
     * "Could not determine number of initially screened papers"
     * "Date range of literature search not explicitly stated"
     * "Only 8 paper titles found, but text mentions 11 included studies"
     * "Cannot generate optimal query - insufficient PICO information"

6. **Research Aim/Objective** (EXTRACT BEFORE CRITERIA):
   - Locate the research aim, objective, or purpose of the systematic review
   - Look for sections titled: "Objective", "Aim", "Purpose", "Research Question", "Background", "Introduction"
   - Extract the main research question or objective statement
   - This typically appears in the introduction or background section
   - Extract verbatim text if it's a single clear statement, or provide a clear summary if described across multiple sentences
   - Examples of what to extract:
     * "To evaluate the effectiveness of surgical versus conservative treatment for hallux valgus"
     * "The aim of this systematic review was to assess the long-term outcomes of ankle arthrodesis"
     * "This review aims to determine the optimal treatment approach for chronic Achilles tendon ruptures"
   - Return null if the research aim cannot be clearly identified

7. **Inclusion/Exclusion Criteria** (CRITICAL - VERBATIM EXTRACTION):
   - Locate the methods/eligibility section of the systematic review
   - Look for sections titled: "Eligibility Criteria", "Inclusion Criteria", "Exclusion Criteria", "Study Selection", "Methods", "Selection Criteria"
   - Extract the EXACT text as written by the original authors - DO NOT paraphrase, summarize, or reword
   - Preserve original formatting:
     * If criteria are listed as bullet points (• or -), preserve that format
     * If criteria are numbered (1., 2., 3.), preserve that format
     * If criteria are in paragraphs, preserve paragraph structure
     * Preserve capitalization, punctuation, and spacing exactly as written
   - For inclusion_criteria: Extract ALL inclusion criteria verbatim
   - For exclusion_criteria: Extract ALL exclusion criteria verbatim
   - If criteria are mixed together, separate them based on context (e.g., "Studies were included if..." vs "Studies were excluded if...")
   - If criteria are not explicitly labeled, infer from context (e.g., "We included studies that..." = inclusion criteria)
   - Return null if criteria section cannot be found or identified
   - Examples of what to extract:
     * "Studies were included if they: (1) enrolled adult patients with hallux valgus, (2) compared surgical to conservative treatment, (3) reported functional outcomes at minimum 6 months follow-up."
     * "Exclusion criteria: (1) pediatric patients, (2) case reports, (3) non-English publications, (4) studies without control groups."
   - DO NOT summarize or rephrase - extract the exact wording from the article

8. **Handle Noise**:
   - Ignore advertisements, navigation menus, author bios, related articles
   - Focus only on systematic review content
   - If multiple studies are discussed, focus on the PRIMARY systematic review

OUTPUT REQUIREMENTS:
- Return null for fields that cannot be determined (never guess)
- Return empty list for paper titles if none found
- Always provide extraction_warnings for missing data
- Ensure paper titles are EXACTLY as written (character-perfect matching)"""


@app.route('/api/extract-systematic-review', methods=['POST'])
def extract_systematic_review():
    """
    Extract key information from systematic review text using GPT-4o-mini.
    
    Request body:
    {
        "text": "Full systematic review webpage content..."
    }
    
    Response:
    {
        "date_range": "2010-2020" or null,
        "initial_papers_found": 523 or null,
        "final_papers_included": 11 or null,
        "included_paper_titles": ["Title 1", "Title 2", ...],
        "short_query": "concise query (10-15 words)" or null,
        "long_query": "comprehensive query (30-50 words)" or null,
        "pico_breakdown": "P: ..., I: ..., C: ..., O: ..." or null,
        "inclusion_criteria": "verbatim inclusion criteria text" or null,
        "exclusion_criteria": "verbatim exclusion criteria text" or null,
        "extraction_warnings": ["warning 1", "warning 2", ...],
        "success": true/false,
        "error": "error message" (if success=false)
    }
    """
    try:
        # Check if OpenAI is configured
        if not openai_client:
            return jsonify({
                "success": False,
                "error": "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
            }), 500
        
        data = request.get_json()
        review_text = data.get('text', '').strip()
        
        if not review_text:
            return jsonify({
                "success": False,
                "error": "No text provided. Please paste systematic review content."
            }), 400
        
        if len(review_text) < 100:
            return jsonify({
                "success": False,
                "error": "Text too short. Please paste the full systematic review content."
            }), 400
        
        # Call OpenAI API with structured output
        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract information from this systematic review text:\n\n{review_text}"}
            ],
            response_format=SystematicReviewExtraction,
            temperature=0.1,  # Low temperature for consistent extraction
        )
        
        # Parse response
        extraction = completion.choices[0].message.parsed
        
        # Build warnings list
        warnings = list(extraction.extraction_warnings) if extraction.extraction_warnings else []
        
        if extraction.date_range is None:
            warnings.append("⚠️ Could not determine date range of literature search")
        
        if extraction.initial_papers_found is None:
            warnings.append("⚠️ Could not determine number of initially screened papers")
        
        if extraction.final_papers_included is None:
            warnings.append("⚠️ Could not determine number of papers included in final analysis")
        
        if not extraction.included_paper_titles:
            warnings.append("⚠️ Could not extract any paper titles from the text")
        elif extraction.final_papers_included and len(extraction.included_paper_titles) != extraction.final_papers_included:
            warnings.append(f"⚠️ Found {len(extraction.included_paper_titles)} paper titles, but text mentions {extraction.final_papers_included} included studies")
        
        if extraction.short_query is None and extraction.long_query is None:
            warnings.append("⚠️ Could not generate search queries - insufficient PICO information")
        elif extraction.short_query is None:
            warnings.append("⚠️ Could not generate short query")
        elif extraction.long_query is None:
            warnings.append("⚠️ Could not generate long query")
        
        if extraction.pico_breakdown is None:
            warnings.append("⚠️ Could not determine PICO breakdown")
        
        if extraction.research_aim is None:
            warnings.append("⚠️ Could not extract research aim/objective")
        
        if extraction.inclusion_criteria is None:
            warnings.append("⚠️ Could not extract inclusion criteria from the text")
        
        if extraction.exclusion_criteria is None:
            warnings.append("⚠️ Could not extract exclusion criteria from the text")
        
        # Clean titles: remove authors, years, journal names, DOIs if accidentally included
        cleaned_titles = []
        for title in extraction.included_paper_titles:
            if title:
                cleaned = title.strip()
                
                # Remove patterns like "Author et al. (Year)" or "Author, A. (Year)" at the start
                # Pattern: Author names followed by year in parentheses
                cleaned = re.sub(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+et\s+al\.?)?\s*[,\.]?\s*\(?\d{4}\)?\s*[:\-\.]?\s*', '', cleaned, flags=re.IGNORECASE)
                # Remove patterns like "Author A, Author B (Year)" at the start
                cleaned = re.sub(r'^[A-Z][a-z]+\s+[A-Z]\.?\s*,\s*[A-Z][a-z]+\s+[A-Z]\.?\s*\(?\d{4}\)?\s*[:\-\.]?\s*', '', cleaned)
                # Remove standalone year in parentheses at start or end
                cleaned = re.sub(r'^\s*\(\d{4}\)\s*', '', cleaned)
                cleaned = re.sub(r'\s*\(\d{4}\)\s*$', '', cleaned)
                # Remove DOI patterns anywhere
                cleaned = re.sub(r'\s*DOI[:\s]*10\.\d+/[^\s]+', '', cleaned, flags=re.IGNORECASE)
                # Remove PMID patterns anywhere
                cleaned = re.sub(r'\s*PMID[:\s]*\d+', '', cleaned, flags=re.IGNORECASE)
                # Remove common journal citation patterns at the end (e.g., ". Journal Name. Year")
                # This pattern: ". Journal Name. Year" at the end
                cleaned = re.sub(r'\.\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\.\s*\d{4}\s*$', '', cleaned)
                # Remove URLs
                cleaned = re.sub(r'\s*https?://[^\s]+', '', cleaned, flags=re.IGNORECASE)
                
                cleaned = cleaned.strip()
                
                # Remove leading/trailing punctuation that might be left over
                cleaned = re.sub(r'^[:\-\.\s]+', '', cleaned)
                cleaned = re.sub(r'[:\-\.\s]+$', '', cleaned)
                cleaned = cleaned.strip()
                
                # Only add if it still looks like a title (has reasonable length and doesn't start with common non-title words)
                if len(cleaned) > 5 and not cleaned.lower().startswith(('doi:', 'pmid:', 'http', 'www.')):
                    cleaned_titles.append(cleaned)
                else:
                    # If cleaning removed too much, use original but log warning
                    logger.warning(f"Title cleaning may have been too aggressive for: {title[:50]}... Using original.")
                    # Try to extract just the title part from the original
                    # Look for text between periods or after common patterns
                    original_cleaned = title.strip()
                    # Remove everything before the first period that might be author
                    if '.' in original_cleaned:
                        parts = original_cleaned.split('.')
                        # Take the part that looks most like a title (longest, doesn't start with numbers/years)
                        for part in parts:
                            part = part.strip()
                            if len(part) > 10 and not part[0].isdigit() and not part.lower().startswith(('doi', 'pmid', 'http')):
                                cleaned_titles.append(part)
                                break
                        else:
                            # If no good part found, use original
                            cleaned_titles.append(title.strip())
                    else:
                        cleaned_titles.append(title.strip())
        
        # Use cleaned titles
        extraction.included_paper_titles = cleaned_titles
        
        logger.info(f"Extracted systematic review: {len(extraction.included_paper_titles)} titles (after cleaning), short_query: {extraction.short_query}, long_query: {extraction.long_query}")
        
        return jsonify({
            "success": True,
            "date_range": extraction.date_range,
            "initial_papers_found": extraction.initial_papers_found,
            "final_papers_included": extraction.final_papers_included,
            "included_paper_titles": extraction.included_paper_titles,
            "short_query": extraction.short_query,
            "long_query": extraction.long_query,
            "pico_breakdown": extraction.pico_breakdown,
            "research_aim": extraction.research_aim,
            "inclusion_criteria": extraction.inclusion_criteria,
            "exclusion_criteria": extraction.exclusion_criteria,
            "extraction_warnings": warnings
        })
    
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in /api/extract-systematic-review: {error_msg}")
        logger.error(error_trace)
        return jsonify({
            "success": False,
            "error": f"Extraction failed: {error_msg}"
        }), 500


# Pydantic models for screening responses
class CriteriaAssessment(BaseModel):
    """Assessment of individual criteria"""
    meets_inclusion: Optional[bool] = None
    violates_exclusion: Optional[bool] = None
    unclear_aspects: Optional[str] = None

class InclusionCriterionCheck(BaseModel):
    """Check for individual inclusion criterion"""
    criterion: str
    met: Optional[bool] = None
    evidence: str

class ExclusionCriterionCheck(BaseModel):
    """Check for individual exclusion criterion"""
    criterion: str
    violated: Optional[bool] = None
    evidence: str

class GPT4oScreeningResponse(BaseModel):
    """Structured response from GPT-4o-mini"""
    decision: str = Field(description="'include', 'exclude', or 'uncertain'")
    confidence: str = Field(description="'high', 'medium', or 'low'")
    reasoning: str = Field(description="Brief explanation of decision (2-3 sentences)")
    criteria_assessment: CriteriaAssessment

class GPT5MiniScreeningResponse(BaseModel):
    """Structured response from GPT-5-mini"""
    decision: str = Field(description="'include', 'exclude', or 'uncertain'")
    confidence: str = Field(description="'high', 'medium', or 'low'")
    reasoning: str = Field(description="Brief explanation of decision (2-3 sentences)")
    criteria_assessment: CriteriaAssessment

class ScreeningResult(BaseModel):
    """Final screening result for a paper"""
    pmid: str
    title: str
    abstract: str
    gpt4o_decision: str
    gpt4o_confidence: str
    gpt4o_reasoning: str
    gpt5mini_decision: str
    gpt5mini_confidence: str
    gpt5mini_reasoning: str
    agreement: bool
    final_category: str  # "include", "exclude", "disagreement", "uncertain"


def create_cache_key(title: str, abstract: str, inclusion_criteria: str, exclusion_criteria: str, research_aim: str = None) -> str:
    """Create a cache key for screening results"""
    content = f"{title}|{abstract}|{inclusion_criteria}|{exclusion_criteria}"
    if research_aim:
        content += f"|{research_aim}"
    return hashlib.md5(content.encode()).hexdigest()


def screen_with_gpt4o(title: str, abstract: str, inclusion_criteria: str, exclusion_criteria: str, research_aim: str = None) -> dict:
    """Screen a paper using OpenAI GPT-4o-mini"""
    if not openai_client:
        raise ValueError("OpenAI client not configured")
    
    # Build research aim section for prompt
    research_aim_section = ""
    if research_aim:
        research_aim_section = f"""
RESEARCH AIM:

{research_aim}

This describes the overall objective and focus of the systematic review. Use this context to better understand what the review is trying to achieve when evaluating eligibility criteria.
"""
    
    prompt = f"""You are an expert systematic review screener. Your task is to determine whether a research article meets the eligibility criteria for inclusion in a systematic review.
{research_aim_section}
INCLUSION CRITERIA:

{inclusion_criteria}

EXCLUSION CRITERIA:

{exclusion_criteria}

ARTICLE TO SCREEN:

Title: {title}

Abstract: {abstract}

INSTRUCTIONS:

1. Carefully read the research aim to understand the systematic review's objective and focus
2. Carefully read the inclusion and exclusion criteria
3. Review the article's title and abstract
4. Determine if the article meets ALL inclusion criteria
5. Determine if the article violates ANY exclusion criteria
6. Consider whether the article aligns with the research aim of the systematic review
7. Make a final decision: INCLUDE or EXCLUDE

IMPORTANT:

- Use the research aim to understand the systematic review's focus and ensure the article aligns with the study's objective
- If information is missing or unclear, indicate uncertainty
- Be conservative: when in doubt, lean toward INCLUDE to avoid missing relevant studies
- Base your decision ONLY on the title and abstract provided
- Do not make assumptions about the full text

Respond in the following JSON format:

{{
  "decision": "include" or "exclude" or "uncertain",
  "confidence": "high" or "medium" or "low",
  "reasoning": "Brief explanation of your decision (2-3 sentences)",
  "criteria_assessment": {{
    "meets_inclusion": true/false,
    "violates_exclusion": true/false,
    "unclear_aspects": "any aspects that are unclear from the abstract"
  }}
}}"""

    try:
        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert systematic review screener. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format=GPT4oScreeningResponse,
            temperature=0.1,
        )
        
        result = completion.choices[0].message.parsed
        return {
            "decision": result.decision.lower(),
            "confidence": result.confidence.lower(),
            "reasoning": result.reasoning,
            "criteria_assessment": {
                "meets_inclusion": result.criteria_assessment.meets_inclusion,
                "violates_exclusion": result.criteria_assessment.violates_exclusion,
                "unclear_aspects": result.criteria_assessment.unclear_aspects
            }
        }
    except Exception as e:
        logger.error(f"Error screening with GPT-4o: {e}")
        raise


def screen_with_gpt5mini(title: str, abstract: str, inclusion_criteria: str, exclusion_criteria: str, research_aim: str = None) -> dict:
    """Screen a paper using OpenAI GPT-5-mini"""
    if not gpt5mini_client:
        raise ValueError("GPT-5-mini client not configured")
    
    # Build research aim section for prompt
    research_aim_section = ""
    if research_aim:
        research_aim_section = f"""
RESEARCH AIM:

{research_aim}

This describes the overall objective and focus of the systematic review. Use this context to better understand what the review is trying to achieve when evaluating eligibility criteria.
"""
    
    prompt = f"""You are an expert systematic review screener. Your task is to determine whether a research article meets the eligibility criteria for inclusion in a systematic review.
{research_aim_section}
INCLUSION CRITERIA:

{inclusion_criteria}

EXCLUSION CRITERIA:

{exclusion_criteria}

ARTICLE TO SCREEN:

Title: {title}

Abstract: {abstract}

INSTRUCTIONS:

1. Carefully read the research aim to understand the systematic review's objective and focus
2. Carefully read the inclusion and exclusion criteria
3. Review the article's title and abstract
4. Determine if the article meets ALL inclusion criteria
5. Determine if the article violates ANY exclusion criteria
6. Consider whether the article aligns with the research aim of the systematic review
7. Make a final decision: INCLUDE or EXCLUDE

IMPORTANT:

- Use the research aim to understand the systematic review's focus and ensure the article aligns with the study's objective
- If information is missing or unclear, indicate uncertainty
- Be conservative: when in doubt, lean toward INCLUDE to avoid missing relevant studies
- Base your decision ONLY on the title and abstract provided
- Do not make assumptions about the full text

Respond in the following JSON format:

{{
  "decision": "include" or "exclude" or "uncertain",
  "confidence": "high" or "medium" or "low",
  "reasoning": "Brief explanation of your decision (2-3 sentences)",
  "criteria_assessment": {{
    "meets_inclusion": true/false,
    "violates_exclusion": true/false,
    "unclear_aspects": "any aspects that are unclear from the abstract"
  }}
}}"""

    try:
        completion = gpt5mini_client.beta.chat.completions.parse(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are an expert systematic review screener. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format=GPT5MiniScreeningResponse,
            # Note: GPT-5-mini only supports default temperature (1.0), cannot set custom temperature
        )
        
        result = completion.choices[0].message.parsed
        return {
            "decision": result.decision.lower(),
            "confidence": result.confidence.lower(),
            "reasoning": result.reasoning,
            "criteria_assessment": {
                "meets_inclusion": result.criteria_assessment.meets_inclusion,
                "violates_exclusion": result.criteria_assessment.violates_exclusion,
                "unclear_aspects": result.criteria_assessment.unclear_aspects
            }
        }
    except Exception as e:
        logger.error(f"Error screening with GPT-5-mini: {e}")
        raise


def screen_single_paper(paper: dict, inclusion_criteria: str, exclusion_criteria: str, research_aim: str = None, use_cache: bool = True) -> dict:
    """Screen a single paper with both LLMs"""
    pmid = paper.get('pmid', '')
    title = paper.get('title', '')
    abstract = paper.get('abstract', '') or ''
    
    # Check cache
    cache_key = create_cache_key(title, abstract, inclusion_criteria, exclusion_criteria, research_aim)
    if use_cache and cache_key in screening_cache:
        logger.debug(f"Using cached result for PMID {pmid}")
        return screening_cache[cache_key]
    
    # Screen with both GPT models in parallel
    gpt4o_result = None
    gpt5mini_result = None
    gpt4o_error = None
    gpt5mini_error = None
    
    # Try GPT-4o-mini with retry
    for attempt in range(2):
        try:
            gpt4o_result = screen_with_gpt4o(title, abstract, inclusion_criteria, exclusion_criteria, research_aim)
            break
        except Exception as e:
            gpt4o_error = str(e)
            if attempt < 1:
                time.sleep(1)  # Brief delay before retry
    
    # Try GPT-5-mini with retry (only if client is available)
    if gpt5mini_client:
        for attempt in range(2):
            try:
                gpt5mini_result = screen_with_gpt5mini(title, abstract, inclusion_criteria, exclusion_criteria, research_aim)
                break
            except Exception as e:
                gpt5mini_error = str(e)
                logger.warning(f"GPT-5-mini screening attempt {attempt + 1} failed for PMID {pmid}: {e}")
                if attempt < 1:
                    time.sleep(1)  # Brief delay before retry
    else:
        gpt5mini_error = "GPT-5-mini client not configured. Check server logs for initialization errors."
        logger.debug(f"GPT-5-mini client not available for PMID {pmid}")
    
    # Determine final category
    if gpt4o_result and gpt5mini_result:
        gpt4o_decision = gpt4o_result['decision']
        gpt5mini_decision = gpt5mini_result['decision']
        
        # Normalize decisions
        if gpt4o_decision in ['include', 'included']:
            gpt4o_decision = 'include'
        elif gpt4o_decision in ['exclude', 'excluded']:
            gpt4o_decision = 'exclude'
        else:
            gpt4o_decision = 'uncertain'
            
        if gpt5mini_decision in ['include', 'included']:
            gpt5mini_decision = 'include'
        elif gpt5mini_decision in ['exclude', 'excluded']:
            gpt5mini_decision = 'exclude'
        else:
            gpt5mini_decision = 'uncertain'
        
        agreement = (gpt4o_decision == gpt5mini_decision) and (gpt4o_decision != 'uncertain')
        
        if agreement:
            final_category = gpt4o_decision
        elif gpt4o_decision == 'uncertain' or gpt5mini_decision == 'uncertain':
            final_category = 'uncertain'
        else:
            final_category = 'disagreement'
    elif gpt4o_result:
        final_category = 'uncertain'  # Only one LLM responded
        gpt5mini_decision = 'error'
        gpt5mini_confidence = 'low'
        gpt5mini_reasoning = f"Error: {gpt5mini_error}" if gpt5mini_error else "No response from GPT-5-mini"
    elif gpt5mini_result:
        final_category = 'uncertain'  # Only one LLM responded
        gpt4o_decision = 'error'
        gpt4o_confidence = 'low'
        gpt4o_reasoning = f"Error: {gpt4o_error}" if gpt4o_error else "No response from GPT-4o-mini"
    else:
        final_category = 'error'
        gpt4o_decision = 'error'
        gpt5mini_decision = 'error'
        gpt4o_reasoning = f"Error: {gpt4o_error}" if gpt4o_error else "No response"
        gpt5mini_reasoning = f"Error: {gpt5mini_error}" if gpt5mini_error else "No response"
    
    result = {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "gpt4o_decision": gpt4o_result['decision'] if gpt4o_result else 'error',
        "gpt4o_confidence": gpt4o_result['confidence'] if gpt4o_result else 'low',
        "gpt4o_reasoning": gpt4o_result['reasoning'] if gpt4o_result else (gpt4o_error or "No response"),
        "gpt5mini_decision": gpt5mini_result['decision'] if gpt5mini_result else 'error',
        "gpt5mini_confidence": gpt5mini_result['confidence'] if gpt5mini_result else 'low',
        "gpt5mini_reasoning": gpt5mini_result['reasoning'] if gpt5mini_result else (gpt5mini_error or "No response"),
        "agreement": agreement if (gpt4o_result and gpt5mini_result) else False,
        "final_category": final_category
    }
    
    # Cache the result
    if use_cache:
        screening_cache[cache_key] = result
    
    return result


@app.route('/api/screen-papers', methods=['POST'])
def screen_papers():
    """
    Screen papers using dual LLM approach (GPT-4o-mini + GPT-5-mini).
    
    Request body:
    {
        "papers": [
            {"pmid": "123", "title": "...", "abstract": "..."},
            ...
        ],
        "inclusion_criteria": "verbatim inclusion criteria text",
        "exclusion_criteria": "verbatim exclusion criteria text",
        "max_workers": 5  # Optional, default 5
    }
    
    Response:
    {
        "success": true,
        "results": [
            {
                "pmid": "123",
                "title": "...",
                "abstract": "...",
                "gpt4o_decision": "include",
                "gpt4o_confidence": "high",
                "gpt4o_reasoning": "...",
                "gpt5mini_decision": "include",
                "gpt5mini_confidence": "medium",
                "gpt5mini_reasoning": "...",
                "agreement": true,
                "final_category": "include"
            },
            ...
        ],
        "summary": {
            "total": 10,
            "include": 3,
            "exclude": 5,
            "disagreement": 1,
            "uncertain": 1
        },
        "errors": []
    }
    """
    try:
        # Check if at least one LLM is configured
        if not openai_client:
            return jsonify({
                "success": False,
                "error": "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
            }), 500
        
        # Warn if GPT-5-mini is not available, but allow screening with GPT-4o-mini only
        if not gpt5mini_client:
            logger.warning("GPT-5-mini client not available - screening will use GPT-4o-mini only (single LLM mode)")
            # Don't fail, just continue with GPT-4o-mini only
        
        data = request.get_json()
        papers = data.get('papers', [])
        inclusion_criteria = data.get('inclusion_criteria', '').strip()
        exclusion_criteria = data.get('exclusion_criteria', '').strip()
        research_aim = data.get('research_aim', '').strip() or None
        max_workers = data.get('max_workers', 5)
        
        if not papers:
            return jsonify({
                "success": False,
                "error": "No papers provided for screening"
            }), 400
        
        if not inclusion_criteria and not exclusion_criteria:
            return jsonify({
                "success": False,
                "error": "Inclusion and/or exclusion criteria must be provided"
            }), 400
        
        logger.info(f"Starting dual LLM screening for {len(papers)} papers")
        
        # Screen papers in parallel
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all screening tasks
            future_to_paper = {
                executor.submit(screen_single_paper, paper, inclusion_criteria, exclusion_criteria, research_aim): paper
                for paper in papers
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error screening paper {paper.get('pmid', 'unknown')}: {e}")
                    errors.append({
                        "pmid": paper.get('pmid', 'unknown'),
                        "title": paper.get('title', ''),
                        "error": str(e)
                    })
        
        # Calculate summary statistics
        summary = {
            "total": len(results),
            "include": sum(1 for r in results if r['final_category'] == 'include'),
            "exclude": sum(1 for r in results if r['final_category'] == 'exclude'),
            "disagreement": sum(1 for r in results if r['final_category'] == 'disagreement'),
            "uncertain": sum(1 for r in results if r['final_category'] in ['uncertain', 'error'])
        }
        
        logger.info(f"Screening complete: {summary}")
        
        return jsonify({
            "success": True,
            "results": results,
            "summary": summary,
            "errors": errors
        })
    
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in /api/screen-papers: {error_msg}")
        logger.error(error_trace)
        return jsonify({
            "success": False,
            "error": f"Screening failed: {error_msg}"
        }), 500


def arbitrate_with_gpt5(
    title: str,
    abstract: str,
    pmid: str,
    inclusion_criteria: str,
    exclusion_criteria: str,
    gpt4o_mini_result: Dict,
    gpt5_mini_result: Dict,
    research_aim: str = None
) -> Dict:
    """
    Use GPT-5 (full model) as supervisor to make final decision.
    This is called for Disagreement and Uncertain cases.
    """
    if not gpt5_client:
        return {
            "final_decision": "exclude",
            "confidence": "low",
            "supervisor_reasoning": "GPT-5 supervisor not available. Defaulting to EXCLUDE for safety.",
            "model": "gpt-5",
            "pmid": pmid,
            "supervision_successful": False,
            "error": "GPT-5 client not configured"
        }
    
    # Build research aim section for prompt
    research_aim_section = ""
    if research_aim:
        research_aim_section = f"""
RESEARCH AIM:

{research_aim}

This describes the overall objective and focus of the systematic review. Use this context to better understand what the review is trying to achieve when evaluating eligibility criteria.
"""
    
    prompt = f"""You are the FINAL SUPERVISOR for a systematic review screening decision.
{research_aim_section}
Two screening models previously reviewed this paper:

- GPT-4o-mini: {gpt4o_mini_result['decision']} (confidence: {gpt4o_mini_result['confidence']})

- GPT-5-mini: {gpt5_mini_result['decision']} (confidence: {gpt5_mini_result['confidence']})

They either disagreed or were uncertain. As the senior supervisor, you must make the FINAL, DEFINITIVE decision.

INCLUSION CRITERIA:

{inclusion_criteria}

EXCLUSION CRITERIA:

{exclusion_criteria}

ARTICLE TO REVIEW:

Title: {title}

Abstract: {abstract}

PREVIOUS REASONING:

GPT-4o-mini: {gpt4o_mini_result.get('reasoning', 'No reasoning provided')}

GPT-5-mini: {gpt5_mini_result.get('reasoning', 'No reasoning provided')}

YOUR TASK AS SUPERVISOR:

Make the FINAL decision: INCLUDE or EXCLUDE (no "uncertain" allowed)

DECISION FRAMEWORK:

1. Review the research aim to understand the systematic review's objective and focus
2. Systematically evaluate each inclusion criterion
3. Systematically evaluate each exclusion criterion  
4. Consider both junior reviewers' reasoning
5. Assess whether the article aligns with the research aim of the systematic review
6. If study meets ALL inclusion AND violates NO exclusion AND aligns with research aim → INCLUDE
7. If ANY doubt or ANY exclusion violated → EXCLUDE (conservative)
8. Base decision ONLY on title and abstract provided

Provide your final judgment with detailed reasoning explaining:

- How the article aligns with the research aim and study context
- Your systematic analysis of each criterion
- Where you agree/disagree with each junior reviewer
- Which criteria were decisive in your final decision
- Your confidence level

Respond in JSON format:

{{
  "final_decision": "include" or "exclude",
  "confidence": "high" or "medium" or "low",
  "supervisor_reasoning": "Detailed 4-6 sentence systematic analysis explaining your decision",
  "agrees_with_4o_mini": true or false,
  "agrees_with_5_mini": true or false,
  "decisive_factors": "Which specific criteria led to your decision",
  "reviewer_analysis": "Brief assessment of why junior reviewers disagreed or were uncertain"
}}"""

    try:
        response = gpt5_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a senior systematic review supervisor with advanced reasoning. Provide thorough, authoritative decisions. Always respond with valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        result["model"] = "gpt-5"
        result["pmid"] = pmid
        result["supervision_successful"] = True
        
        logger.info(f"✅ GPT-5 Supervisor decided: {result.get('final_decision', 'unknown').upper()} for PMID {pmid}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ GPT-5 supervision error for PMID {pmid}: {str(e)}")
        return {
            "final_decision": "exclude",
            "confidence": "low",
            "supervisor_reasoning": f"Supervision failed: {str(e)}. Defaulting to EXCLUDE for safety.",
            "model": "gpt-5",
            "pmid": pmid,
            "supervision_successful": False,
            "error": str(e)
        }


@app.route('/api/arbitrate-paper', methods=['POST'])
def arbitrate_paper():
    """
    Use GPT-5 as supervisor to make final decision for a paper with disagreement or uncertainty.
    
    Request body:
    {
        "pmid": "123",
        "title": "...",
        "abstract": "...",
        "inclusion_criteria": "...",
        "exclusion_criteria": "...",
        "gpt4o_mini_result": {
            "decision": "include",
            "confidence": "high",
            "reasoning": "..."
        },
        "gpt5_mini_result": {
            "decision": "exclude",
            "confidence": "medium",
            "reasoning": "..."
        }
    }
    
    Response:
    {
        "success": true,
        "pmid": "123",
        "final_decision": "include",
        "confidence": "high",
        "supervisor_reasoning": "...",
        "agrees_with_4o_mini": true,
        "agrees_with_5_mini": false,
        "decisive_factors": "...",
        "reviewer_analysis": "...",
        "supervision_successful": true
    }
    """
    try:
        if not gpt5_client:
            return jsonify({
                "success": False,
                "error": "GPT-5 supervisor not configured. Please check OpenAI API key."
            }), 500
        
        data = request.get_json()
        pmid = data.get('pmid')
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        inclusion_criteria = data.get('inclusion_criteria', '').strip()
        exclusion_criteria = data.get('exclusion_criteria', '').strip()
        research_aim = data.get('research_aim', '').strip() or None
        gpt4o_mini_result = data.get('gpt4o_mini_result', {})
        gpt5_mini_result = data.get('gpt5_mini_result', {})
        
        if not pmid:
            return jsonify({
                "success": False,
                "error": "PMID is required"
            }), 400
        
        if not inclusion_criteria and not exclusion_criteria:
            return jsonify({
                "success": False,
                "error": "Inclusion and/or exclusion criteria must be provided"
            }), 400
        
        logger.info(f"Starting GPT-5 arbitration for PMID {pmid}")
        
        result = arbitrate_with_gpt5(
            title=title,
            abstract=abstract,
            pmid=pmid,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            gpt4o_mini_result=gpt4o_mini_result,
            gpt5_mini_result=gpt5_mini_result,
            research_aim=research_aim
        )
        
        return jsonify({
            "success": True,
            **result
        })
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in /api/arbitrate-paper: {error_msg}")
        logger.error(error_trace)
        return jsonify({
            "success": False,
            "error": f"Arbitration failed: {error_msg}"
        }), 500


@app.route('/api/generate-search-query', methods=['POST'])
def api_generate_search_query():
    """Generate optimized semantic search query using GPT-4o-mini based on PICO and criteria"""
    try:
        data = request.get_json()
        population = data.get('population', '')
        intervention = data.get('intervention', '')
        comparison = data.get('comparison', '')
        outcome = data.get('outcome', '')
        research_question = data.get('research_question', '')
        inclusion_criteria = data.get('inclusion_criteria', '')
        exclusion_criteria = data.get('exclusion_criteria', '')
        
        if not openai_client:
            return jsonify({"error": "OpenAI client not configured"}), 500
        
        # Build prompt for query generation
        prompt = f"""You are an expert in systematic review search strategy development. Your task is to generate an optimized semantic search query for finding relevant research articles.

PICO FRAMEWORK:
- Population: {population if population else 'Not specified'}
- Intervention: {intervention if intervention else 'Not specified'}
- Comparison: {comparison if comparison else 'Not specified'}
- Outcome: {outcome if outcome else 'Not specified'}

RESEARCH QUESTION:
{research_question if research_question else 'Not specified'}

INCLUSION CRITERIA:
{inclusion_criteria if inclusion_criteria else 'Not specified'}

EXCLUSION CRITERIA:
{exclusion_criteria if exclusion_criteria else 'Not specified'}

TASK:
Generate a comprehensive semantic search query that will effectively retrieve articles relevant to this systematic review. The query should:
1. Capture the key concepts from the PICO framework
2. Include relevant synonyms and related terms
3. Focus on the research question and inclusion criteria
4. Be optimized for semantic/vector search (not PubMed syntax)

The query should be a natural language description (2-4 sentences) that captures the essence of what we're looking for. It should be comprehensive enough to retrieve relevant papers but focused enough to avoid too many irrelevant results.

Respond with ONLY the search query text, no additional explanation or formatting."""

        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in systematic review search strategy. Generate optimized semantic search queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            generated_query = completion.choices[0].message.content.strip()
            
            logger.info(f"Generated search query: {generated_query[:100]}...")
            
            return jsonify({
                "success": True,
                "query": generated_query
            })
        except Exception as e:
            logger.error(f"Error generating search query: {e}")
            return jsonify({
                "success": False,
                "error": f"Failed to generate search query: {str(e)}"
            }), 500
            
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error in /api/generate-search-query: {error_msg}")
        logger.error(error_trace)
        return jsonify({
            "success": False,
            "error": f"Request failed: {error_msg}"
        }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("SEMANTIC SEARCH ENGINE - WEB INTERFACE")
    print("="*60)
    print(f"\nStarting server at http://{FLASK_HOST}:{FLASK_PORT}")
    print("Press Ctrl+C to stop\n")
    print("Note: Search engine will initialize on first search request")
    print("Loading cross-encoder reranker (best model)...")
    try:
        reranker = get_reranker("best")
        print("✓ Reranker ready\n")
    except Exception as e:
        logger.warning(f"Could not preload reranker: {e}")
        print("⚠ Reranker will load on first use\n")
    
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG_MODE)

