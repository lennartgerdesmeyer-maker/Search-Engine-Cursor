#!/usr/bin/env python3
"""
QUICK REBUILD - Complete system reset and rebuild
Run this script to completely rebuild your semantic search system from scratch.
"""
import sys
import os
import json
import signal
import sqlite3
import subprocess
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config.config import DATA_DIR, LOG_DIR, METADATA_DB_PATH
from src.embedding_generator import EmbeddingGenerator
from src.vector_index import VectorIndex

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def print_step(step_num, total_steps, text):
    """Print a formatted step"""
    print(f"\n[STEP {step_num}/{total_steps}] {text}")
    print("-" * 80)

def kill_stuck_processes():
    """Kill any stuck embedding processes"""
    print("Checking for stuck processes...")

    try:
        # Find Python processes related to embeddings
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )

        killed_any = False
        for line in result.stdout.split('\n'):
            if 'python' in line.lower() and any(keyword in line.lower() for keyword in
                ['embedding', 'generate_embeddings', 'build_complete', 'hard_reset']):
                if 'grep' not in line and 'quick_rebuild' not in line:
                    try:
                        pid = int(line.split()[1])
                        print(f"  Killing stuck process (PID: {pid})")
                        os.kill(pid, signal.SIGKILL)
                        killed_any = True
                    except (IndexError, ValueError, ProcessLookupError):
                        pass

        if not killed_any:
            print("  ✓ No stuck processes found")
    except Exception as e:
        print(f"  Warning: Could not check for stuck processes: {e}")

def cleanup_old_files():
    """Remove all old embedding and index files"""
    print("Cleaning up old files...")

    files_to_delete = [
        DATA_DIR / "embeddings.npy",
        DATA_DIR / "pmid_index.json",
        DATA_DIR / "faiss_index.bin",
        DATA_DIR / "embedding_mean.npy",
        DATA_DIR / "embedding_checkpoint.json",
        DATA_DIR / "vector_store",
        LOG_DIR / "embedding_generation.log",
        LOG_DIR / "vector_index.log",
    ]

    for file_path in files_to_delete:
        if file_path.exists():
            if file_path.is_dir():
                import shutil
                shutil.rmtree(file_path)
                print(f"  Deleted directory: {file_path.name}")
            else:
                file_path.unlink()
                print(f"  Deleted file: {file_path.name}")

    print("  ✓ All old files cleaned up")

def check_database():
    """Check if database exists and return article count"""
    print("Checking database...")

    if not METADATA_DB_PATH.exists():
        print(f"  ✗ Database NOT found at: {METADATA_DB_PATH}")
        print("\n  ACTION REQUIRED:")
        print("  You need to either:")
        print("    1. Copy your existing metadata.db to the data/ directory, OR")
        print("    2. Download PubMed articles first using:")
        print(f"       python3 src/pubmed_downloader.py")
        print("\n  After that, run this script again.")
        return False

    try:
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM articles")
        count = cursor.fetchone()[0]
        conn.close()

        print(f"  ✓ Database found: {METADATA_DB_PATH}")
        print(f"  ✓ Articles in database: {count:,}")
        return True
    except Exception as e:
        print(f"  ✗ Error reading database: {e}")
        return False

def generate_embeddings():
    """Generate embeddings from scratch"""
    print("Generating embeddings from scratch...")
    print("This may take several hours for large databases (300k+ articles)")
    print("Progress will be saved every 5,000 articles")
    print("")

    try:
        generator = EmbeddingGenerator()
        generator.generate_all_embeddings(resume=False)
        print("\n  ✓ Embedding generation complete")
        return True
    except Exception as e:
        print(f"\n  ✗ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def build_faiss_index():
    """Build FAISS index and calculate mean vector"""
    print("Building FAISS index...")

    try:
        # Calculate embedding mean
        print("  Calculating embedding mean...")
        embeddings = np.load(DATA_DIR / "embeddings.npy")
        embedding_mean = np.mean(embeddings, axis=0)
        np.save(DATA_DIR / "embedding_mean.npy", embedding_mean)
        print(f"  ✓ Mean vector saved (shape: {embedding_mean.shape})")

        # Build FAISS index
        print("  Building FAISS index...")
        vector_index = VectorIndex()
        vector_index.build_index(force_rebuild=True)
        print("  ✓ FAISS index built successfully")
        return True
    except Exception as e:
        print(f"\n  ✗ FAISS index build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_system():
    """Verify that all components are working"""
    print("Verifying system...")

    try:
        # Load and check embeddings
        embeddings = np.load(DATA_DIR / "embeddings.npy")

        # Load and check PMID index
        with open(DATA_DIR / "pmid_index.json", 'r') as f:
            pmids = json.load(f)

        # Check FAISS index
        faiss_size = (DATA_DIR / "faiss_index.bin").stat().st_size / (1024 * 1024)

        # Check embedding mean
        embedding_mean = np.load(DATA_DIR / "embedding_mean.npy")

        print("\n  System Status:")
        print(f"    Embeddings:      {len(embeddings):,} vectors")
        print(f"    PMIDs:           {len(pmids):,} articles")
        print(f"    FAISS index:     {faiss_size:.1f} MB")
        print(f"    Embedding dim:   {embeddings.shape[1]}")
        print(f"    Mean vector:     {embedding_mean.shape}")

        # Check consistency
        if len(embeddings) != len(pmids):
            print(f"\n  ⚠ Warning: Mismatch between embeddings ({len(embeddings)}) and PMIDs ({len(pmids)})")
            return False

        print("\n  ✓ All components verified successfully")
        return True
    except Exception as e:
        print(f"\n  ✗ Verification failed: {e}")
        return False

def main():
    """Main execution function"""
    print_header("COMPLETE FRESH START - Semantic Search Engine Reset")

    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    # Step 1: Kill stuck processes
    print_step(1, 5, "Kill Stuck Processes")
    kill_stuck_processes()

    # Step 2: Clean up old files
    print_step(2, 5, "Clean Up Old Files")
    cleanup_old_files()

    # Step 3: Check database
    print_step(3, 5, "Check Database")
    if not check_database():
        print("\n❌ Cannot proceed without database. Exiting.")
        sys.exit(1)

    # Step 4: Generate embeddings
    print_step(4, 5, "Generate Embeddings")
    if not generate_embeddings():
        print("\n❌ Embedding generation failed. Exiting.")
        sys.exit(1)

    # Step 5: Build FAISS index
    print_step(5, 5, "Build FAISS Index")
    if not build_faiss_index():
        print("\n❌ FAISS index build failed. Exiting.")
        sys.exit(1)

    # Final verification
    print_header("Final Verification")
    if not verify_system():
        print("\n⚠ System verification failed - please check the errors above.")
        sys.exit(1)

    # Success!
    print_header("✓ COMPLETE FRESH START SUCCESSFUL!")
    print("Your semantic search engine is ready to use!")
    print("\nNext steps:")
    print("  • Start the app: python3 start_app_simple.py")
    print("  • Or test a search:")
    print('    python3 -c "from src.search_engine import SearchEngine; '
          'engine = SearchEngine(); '
          'results = engine.search(\'achilles tendon rupture treatment\'); '
          'print(f\'Found {len(results)} results\')"')
    print("")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
