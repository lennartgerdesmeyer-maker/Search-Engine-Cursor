"""
Find PMIDs and Add Missing Papers
Searches PubMed by title or DOI, finds PMIDs, and adds papers to the database.
"""
import sys
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from Bio import Entrez

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.pubmed_downloader import PubMedDownloader, add_custom_articles
from src.embedding_generator import EmbeddingGenerator
from src.vector_index import VectorIndex
from config.config import NCBI_EMAIL, NCBI_API_KEY

# Configure Entrez
Entrez.email = NCBI_EMAIL
Entrez.api_key = NCBI_API_KEY


def doi_to_pmid(doi: str) -> Optional[str]:
    """
    Convert DOI to PMID using NCBI ID Converter API.
    
    Args:
        doi: DOI string (with or without 'doi:' prefix)
    
    Returns:
        PMID string or None if not found
    """
    # Remove 'doi:' prefix if present
    doi_clean = doi.replace('doi:', '').strip()
    
    try:
        url = f"https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids={doi_clean}&format=json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data.get('records') and len(data['records']) > 0:
            pmid = data['records'][0].get('pmid')
            if pmid:
                return str(pmid)
    except Exception as e:
        print(f"  ⚠ Error converting DOI {doi_clean}: {e}")
    
    return None


def search_pubmed_by_title(title: str, max_results: int = 5) -> List[Tuple[str, str]]:
    """
    Search PubMed by title and return list of (PMID, matched_title) tuples.
    
    Args:
        title: Article title to search
        max_results: Maximum number of results to return
    
    Returns:
        List of (PMID, matched_title) tuples
    """
    try:
        # Clean title for search (remove special characters, limit length)
        search_title = title[:200]  # PubMed has query length limits
        
        handle = Entrez.esearch(
            db="pubmed",
            term=f'"{search_title}"[Title]',
            retmax=max_results,
            retmode="xml"
        )
        results = Entrez.read(handle)
        handle.close()
        
        pmids = results.get("IdList", [])
        if not pmids:
            return []
        
        # Fetch titles to verify matches
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(pmids),
            rettype="xml",
            retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()
        
        matches = []
        for record in records.get("PubmedArticle", []):
            try:
                medline = record.get("MedlineCitation", {})
                article = medline.get("Article", {})
                matched_title = article.get("ArticleTitle", "")
                pmid = str(medline.get("PMID", ""))
                
                if pmid and matched_title:
                    matches.append((pmid, matched_title))
            except:
                continue
        
        time.sleep(0.1)  # Rate limiting
        return matches
        
    except Exception as e:
        print(f"  ⚠ Error searching PubMed for '{title[:50]}...': {e}")
        return []


def find_pmids_for_papers(papers: List[Dict[str, str]]) -> Dict[str, Optional[str]]:
    """
    Find PMIDs for a list of papers using title or DOI.
    
    Args:
        papers: List of dicts with 'title' and optionally 'doi' or 'pmid' keys
    
    Returns:
        Dict mapping paper identifier to PMID (or None if not found)
    """
    results = {}
    
    print("\n" + "="*60)
    print("FINDING PMIDs")
    print("="*60)
    
    for i, paper in enumerate(papers, 1):
        title = paper.get('title', '').strip()
        doi = paper.get('doi', '').strip()
        pmid_direct = paper.get('pmid', '').strip()
        identifier = paper.get('identifier', f"Paper {i}")
        
        print(f"\n[{i}/{len(papers)}] {identifier}")
        print(f"  Title: {title[:80]}...")
        
        pmid = None
        
        # Check if PMID is directly provided
        if pmid_direct:
            print(f"  ✓ Using provided PMID: {pmid_direct}")
            results[identifier] = pmid_direct
            continue
        
        # Try DOI first (most reliable)
        if doi:
            print(f"  Trying DOI: {doi}")
            pmid = doi_to_pmid(doi)
            if pmid:
                print(f"  ✓ Found PMID via DOI: {pmid}")
                results[identifier] = pmid
                continue
        
        # Try title search
        if title:
            print(f"  Searching PubMed by title...")
            matches = search_pubmed_by_title(title, max_results=3)
            
            if matches:
                # Check if first match is very similar
                matched_pmid, matched_title = matches[0]
                
                # Simple similarity check (case-insensitive, normalized)
                title_lower = title.lower().strip()
                matched_lower = matched_title.lower().strip()
                
                if title_lower == matched_lower or title_lower in matched_lower or matched_lower in title_lower:
                    pmid = matched_pmid
                    print(f"  ✓ Found PMID via title: {pmid}")
                    print(f"    Matched: {matched_title[:80]}...")
                else:
                    print(f"  ⚠ Found {len(matches)} potential matches, but titles don't match closely:")
                    for idx, (pmid_candidate, match_title) in enumerate(matches[:2], 1):
                        print(f"    {idx}. PMID {pmid_candidate}: {match_title[:60]}...")
                    print(f"  → Skipping (manual review needed)")
        
        if not pmid:
            print(f"  ❌ Could not find PMID")
        
        results[identifier] = pmid
        time.sleep(0.2)  # Rate limiting
    
    return results


def add_papers_and_update_index(pmids: List[str]):
    """Complete workflow to add papers and update the search index."""
    print("\n" + "="*60)
    print("ADDING PAPERS TO DATABASE")
    print("="*60)
    print(f"\nTotal PMIDs to process: {len(pmids)}")
    
    # Step 1: Add papers to database
    print("\n[1/3] Fetching papers from PubMed and adding to database...")
    add_custom_articles(pmids)
    print("✓ Papers added to database")
    
    # Step 2: Generate embeddings for new articles
    print("\n[2/3] Generating embeddings for new articles...")
    generator = EmbeddingGenerator()
    generator.generate_all_embeddings(resume=True)
    print("✓ Embeddings generated")
    
    # Step 3: Rebuild vector index
    print("\n[3/3] Rebuilding vector index...")
    index = VectorIndex()
    index.build_index(force_rebuild=True)
    print("✓ Vector index rebuilt")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    
    stats = index.get_stats()
    print(f"\nTotal vectors in index: {stats['total_vectors']:,}")
    print(f"Index size: {stats['index_size_mb']:.1f} MB")


# Papers without PMIDs from your table
PAPERS_WITHOUT_PMID = [
    {
        'identifier': 'Diabetic Foot - 5-year follow-up',
        'title': 'Five-year follow-up analysis of recurrence and death after ulcer healing in patients with diabetic foot',
        'doi': '10.3760/cma.j.cn115791-20200707-00422'
    },
    {
        'identifier': 'Diabetic Foot - Mortality type 1/2',
        'title': 'Mortality in type 1 diabetes mellitus and type 2 diabetes mellitus: foot ulcer location is an independent risk determinant',
        'doi': '',  # Has PMID: 33772856 (will be added to EXISTING_PMIDS)
        'pmid': '33772856'  # Direct PMID provided
    },
    {
        'identifier': 'Diabetic Foot - Socioeconomic',
        'title': 'Association between socioeconomic position and diabetic foot ulcer outcomes',
        'doi': '',  # Has PMID: 34261483 (will be added to EXISTING_PMIDS)
        'pmid': '34261483'  # Direct PMID provided
    },
    {
        'identifier': 'Achilles - Spanish',
        'title': 'Reparación de la ruptura aguda del tendón calcáneo: estudio comparativo entre dos técnicas quirúrgicas',
        'doi': ''  # No DOI/PMID found yet
    },
    {
        'identifier': 'Achilles - German',
        'title': 'Die frische Achillessehnenruptur: eine prospektive Untersuchung zur Beurteilung verschiedener Therapiemöglichkeiten',
        'doi': '10.1007/PL00003753'
    },
    {
        'identifier': 'Diabetic Foot - Exercise',
        'title': 'Effect of Twelve Weeks Supervised Aerobic Exercise on Ulcer Healing and Changes in Selected Biochemical Profiles of Diabetic Foot Ulcer Subject',
        'doi': '10.5923/j.diabetes.20140303.03'
    },
]

# All PMIDs from your table (already extracted)
EXISTING_PMIDS = [
    "31943705", "27585063", "32209136", "31613404", "32994867", "17555582",
    "27733354", "27993523", "35255070", "35213066", "35658957", "34024706",
    "34913257", "33872634", "33497545", "34874944", "32249374", "32961974",
    "33160884", "31316149", "30606164", "28424173", "26666583", "27118161",
    "25048499", "24865783", "22057196", "21480971", "33996886", "33966109",
    "32165163", "19561389", "3399515", "10597811", "5437126", "15293489",
    "21091133", "27265095", "22980576", "12625653", "8842827", "10471436",
    "20975524", "16719030", "19665825", "11237162", "19254614", "15704507",
    "29865941", "25184801", "19439137", "18066530", "24190345", "23250350",
    "11503980", "31548150", "31679677", "29528724", "33184442", "23085538",
    "20015581", "31393860", "33806449", "25457972", "8691896", "20538390",
    "28356626", "32224985", "25683316", "31079137", "28939307", "20547668",
    "18390492", "29332470", "16476915", "24790486", "29273936", "26378030",
    "19066253", "21030727", "29521922", "34338076", "25212527", "29940667",
    "23064082", "19643657", "31827363", "24879018", "23770660", "22441228",
    "27802962", "31892510", "19329789", "24923269", "26513392", "19042130",
    "20152755", "32197661", "32207336", "31535563", "30448373", "28394631",
    "29409264", "27623732", "26467354", "31608832", "29364026", "23106289",
    "24707086", "27942135", "23632367", "31590069", "32645830", "33202893",
    "27336689", "32911733", "30719446", "33156692", "31962190", "30388670",
    "30860412",
    # Additional PMIDs from user's list
    "33772856",  # Schofield 2021 - Mortality in type 1/2 diabetes
    "34261483",  # Ha 2021 - Association between socioeconomic position
    "33909492",  # Aragón 2021 - Long-term Mortality
    "31400509",  # Amadou 2020 - Five-year mortality
    "29176889",  # Al-Rubeaan 2017 - All-cause mortality
    "34256628",  # Troisi 2021 - Influence of pedal arch quality
    "32820699",  # Piaggesi 2021 - Diabetic foot surgery "made in Italy"
]


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FIND PMIDs AND ADD MISSING PAPERS")
    print("="*60)
    
    # Step 1: Find PMIDs for papers without them
    print("\n" + "="*60)
    print("STEP 1: Finding PMIDs for papers without them")
    print("="*60)
    
    # You can add DOIs here if you have them
    # Example:
    # PAPERS_WITHOUT_PMID[0]['doi'] = '10.1234/example.doi'
    
    # First, collect any direct PMIDs from PAPERS_WITHOUT_PMID
    all_pmids = EXISTING_PMIDS.copy()
    direct_pmids = []
    for paper in PAPERS_WITHOUT_PMID:
        if paper.get('pmid'):
            direct_pmids.append(paper['pmid'])
            if paper['pmid'] not in all_pmids:
                all_pmids.append(paper['pmid'])
    
    if direct_pmids:
        print(f"\nFound {len(direct_pmids)} direct PMIDs in papers list: {', '.join(direct_pmids)}")
    
    found_pmids = find_pmids_for_papers(PAPERS_WITHOUT_PMID)
    
    # Collect all PMIDs (existing + newly found)
    print("\n" + "="*60)
    print("SUMMARY OF PMID SEARCH")
    print("="*60)
    found_count = 0
    for identifier, pmid in found_pmids.items():
        if pmid:
            print(f"✓ {identifier}: PMID {pmid}")
            if pmid not in all_pmids:
                all_pmids.append(pmid)
                found_count += 1
        else:
            print(f"❌ {identifier}: Not found")
    
    print(f"\nFound {found_count} new PMIDs")
    print(f"Total PMIDs to add: {len(all_pmids)}")
    
    # Step 2: Add all papers (automatically proceed)
    print("\n" + "="*60)
    print("AUTOMATICALLY PROCEEDING WITH ADDING PAPERS")
    print("="*60)
    add_papers_and_update_index(all_pmids)

