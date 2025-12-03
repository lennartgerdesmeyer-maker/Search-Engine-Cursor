#!/bin/bash

################################################################################
# COMPLETE FRESH START FOR SEMANTIC SEARCH ENGINE
# This script will:
# 1. Kill any stuck embedding/Python processes
# 2. Clean up ALL old embedding/index files
# 3. Rebuild everything from scratch (embeddings + FAISS index)
################################################################################

echo "════════════════════════════════════════════════════════════════════════════"
echo "  COMPLETE FRESH START - Semantic Search Engine Reset"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# STEP 1: Kill any stuck processes
################################################################################
echo -e "${BLUE}[STEP 1/5]${NC} Checking for stuck processes..."
echo ""

# Find and kill any Python processes related to embeddings
STUCK_PROCESSES=$(ps aux | grep -E "(embedding|generate_embeddings|build_complete|hard_reset)" | grep python | grep -v grep | awk '{print $2}')

if [ -z "$STUCK_PROCESSES" ]; then
    echo -e "${GREEN}✓${NC} No stuck processes found"
else
    echo -e "${YELLOW}⚠${NC} Found stuck processes:"
    ps aux | grep -E "(embedding|generate_embeddings|build_complete|hard_reset)" | grep python | grep -v grep
    echo ""
    echo "Killing stuck processes..."
    for PID in $STUCK_PROCESSES; do
        echo "  Killing PID: $PID"
        kill -9 $PID 2>/dev/null
    done
    echo -e "${GREEN}✓${NC} All stuck processes terminated"
fi
echo ""

################################################################################
# STEP 2: Clean up ALL old files
################################################################################
echo -e "${BLUE}[STEP 2/5]${NC} Cleaning up old files..."
echo ""

# Create data directory if it doesn't exist
mkdir -p data
mkdir -p logs

# List of files to delete
FILES_TO_DELETE=(
    "data/embeddings.npy"
    "data/pmid_index.json"
    "data/faiss_index.bin"
    "data/embedding_mean.npy"
    "data/embedding_checkpoint.json"
    "data/vector_store"
    "logs/embedding_generation.log"
    "logs/vector_index.log"
)

for FILE in "${FILES_TO_DELETE[@]}"; do
    if [ -e "$FILE" ]; then
        echo "  Deleting: $FILE"
        rm -rf "$FILE"
    fi
done

echo -e "${GREEN}✓${NC} All old embedding/index files deleted"
echo ""

################################################################################
# STEP 3: Check for database
################################################################################
echo -e "${BLUE}[STEP 3/5]${NC} Checking for database..."
echo ""

if [ ! -f "data/metadata.db" ]; then
    echo -e "${RED}✗${NC} Database NOT found at: data/metadata.db"
    echo ""
    echo -e "${YELLOW}ACTION REQUIRED:${NC}"
    echo "  You need to either:"
    echo "  1. Copy your existing metadata.db to the data/ directory, OR"
    echo "  2. Download PubMed articles first using:"
    echo ""
    echo -e "     ${GREEN}python3 src/pubmed_downloader.py${NC}"
    echo ""
    echo "  After that, run this script again."
    echo ""
    exit 1
else
    # Check number of articles in database
    ARTICLE_COUNT=$(sqlite3 data/metadata.db "SELECT COUNT(*) FROM articles;" 2>/dev/null)
    echo -e "${GREEN}✓${NC} Database found: data/metadata.db"
    echo "  Articles in database: ${ARTICLE_COUNT}"
fi
echo ""

################################################################################
# STEP 4: Generate embeddings from scratch
################################################################################
echo -e "${BLUE}[STEP 4/5]${NC} Generating embeddings..."
echo ""
echo "This will take a while (several hours for 300k+ articles)..."
echo "Progress will be saved every 5,000 articles."
echo ""

python3 generate_embeddings.py

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}✗${NC} Embedding generation failed!"
    exit 1
fi

echo ""
echo -e "${GREEN}✓${NC} Embedding generation complete"
echo ""

################################################################################
# STEP 5: Build FAISS index
################################################################################
echo -e "${BLUE}[STEP 5/5]${NC} Building FAISS index..."
echo ""

python3 -c "
import sys
from pathlib import Path
sys.path.append(str(Path('.')))
from src.vector_index import VectorIndex
from config.config import DATA_DIR
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# Calculate embedding mean
print('Calculating embedding mean...')
embeddings = np.load(DATA_DIR / 'embeddings.npy')
embedding_mean = np.mean(embeddings, axis=0)
np.save(DATA_DIR / 'embedding_mean.npy', embedding_mean)
print(f'✓ Mean vector saved: shape {embedding_mean.shape}')

# Build FAISS index
print('Building FAISS index...')
vector_index = VectorIndex()
vector_index.build_index(force_rebuild=True)
print('✓ FAISS index built successfully')
"

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}✗${NC} FAISS index build failed!"
    exit 1
fi

echo ""
echo -e "${GREEN}✓${NC} FAISS index build complete"
echo ""

################################################################################
# FINAL VERIFICATION
################################################################################
echo "════════════════════════════════════════════════════════════════════════════"
echo -e "  ${GREEN}✓ COMPLETE FRESH START SUCCESSFUL!${NC}"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "System Status:"
python3 -c "
import sys
from pathlib import Path
sys.path.append(str(Path('.')))
from config.config import DATA_DIR
import json
import numpy as np
import os

# Check files
embeddings = np.load(DATA_DIR / 'embeddings.npy')
with open(DATA_DIR / 'pmid_index.json', 'r') as f:
    pmids = json.load(f)
faiss_size = os.path.getsize(DATA_DIR / 'faiss_index.bin') / (1024 * 1024)

print(f'  Embeddings:     {len(embeddings):,} vectors')
print(f'  PMIDs:          {len(pmids):,} articles')
print(f'  FAISS index:    {faiss_size:.1f} MB')
print(f'  Embedding dim:  {embeddings.shape[1]}')
"
echo ""
echo "Ready to use! Start the app with:"
echo -e "  ${GREEN}python3 start_app_simple.py${NC}"
echo ""
echo "Or test a search with:"
echo -e "  ${GREEN}python3 -c \"from src.search_engine import SearchEngine; engine = SearchEngine(); results = engine.search('achilles tendon rupture treatment'); print(f'Found {len(results)} results')\"${NC}"
echo ""
