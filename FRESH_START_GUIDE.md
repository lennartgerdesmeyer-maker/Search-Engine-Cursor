# 🔄 Complete Fresh Start Guide

Your semantic search engine had issues after manually adding a paper. This guide provides **terminal commands to completely reset and rebuild everything from scratch**.

---

## ✅ Quick Status Check

The good news: **No stuck processes are currently running!** Your system is in a clean state and ready for a fresh rebuild.

---

## 🚀 Three Ways to Rebuild

### Option 1: Python Script (RECOMMENDED) ⭐

**Single command to do everything:**
```bash
python3 quick_rebuild.py
```

**What it does:**
1. ✓ Kills any stuck processes
2. ✓ Deletes all old embeddings/indexes/checkpoints
3. ✓ Checks if your database exists
4. ✓ Generates fresh embeddings (with progress tracking)
5. ✓ Builds new FAISS index
6. ✓ Verifies everything is working

**Advantages:**
- Shows clear progress messages
- Better error handling
- Automatic verification
- Resumable if interrupted

---

### Option 2: Bash Script

**Single command:**
```bash
./complete_fresh_start.sh
```

Same functionality as Option 1, but using bash.

---

### Option 3: Manual Step-by-Step

If you want full control, run these commands one at a time:

```bash
# 1. Kill any stuck processes
pkill -9 -f "embedding|generate_embeddings|build_complete"

# 2. Clean up old files
rm -rf data/embeddings.npy \
       data/pmid_index.json \
       data/faiss_index.bin \
       data/embedding_mean.npy \
       data/embedding_checkpoint.json \
       logs/*.log

# 3. Generate embeddings
python3 generate_embeddings.py

# 4. Build FAISS index
python3 -c "
from src.vector_index import VectorIndex
from config.config import DATA_DIR
import numpy as np

# Calculate mean
embeddings = np.load(DATA_DIR / 'embeddings.npy')
embedding_mean = np.mean(embeddings, axis=0)
np.save(DATA_DIR / 'embedding_mean.npy', embedding_mean)

# Build index
vector_index = VectorIndex()
vector_index.build_index(force_rebuild=True)
print('Done!')
"
```

---

## 📋 Prerequisites

### You Need a Database First!

The scripts will check if `data/metadata.db` exists. If not, you have two options:

**Option A: Copy Your Existing Database**
```bash
# If you have the database file somewhere else:
cp /path/to/your/metadata.db data/metadata.db
```

**Option B: Download Fresh Data from PubMed**
```bash
python3 src/pubmed_downloader.py
```

---

## ⏱️ How Long Will This Take?

- **Database with 315,638 articles**: ~3-6 hours
- **Progress saved every 5,000 articles** (resumable if interrupted)
- **FAISS index building**: ~2-5 minutes

**Factors affecting speed:**
- CPU speed
- GPU availability (MPS on Mac/CUDA on Linux)
- Disk I/O speed

---

## 🔍 Monitor Progress

While embedding generation is running, you can monitor progress in another terminal:

```bash
# Watch the checkpoint file
watch -n 5 'cat data/embedding_checkpoint.json 2>/dev/null'

# Or check the log
tail -f logs/embedding_generation.log

# Or use the progress checker
python3 check_progress.py
```

---

## 🛑 If It Gets Stuck Again

If the process appears stuck (no progress for 10+ minutes):

**1. Check if it's actually stuck:**
```bash
# Check CPU usage (should be >50% if working)
top -p $(pgrep -f embedding)

# Check if checkpoint file is updating
stat data/embedding_checkpoint.json
```

**2. Force kill everything:**
```bash
# Kill all Python processes
pkill -9 python3

# Kill specific embedding processes
pkill -9 -f "embedding|generate_embeddings"

# Nuclear option - kill all Python
killall -9 python
```

**3. Clean up and restart:**
```bash
rm -rf data/embedding_checkpoint.json
python3 quick_rebuild.py
```

---

## ✅ After Successful Rebuild

Once complete, test your system:

**1. Verify files exist:**
```bash
ls -lh data/
# Should see:
# - embeddings.npy (~1GB for 300k articles)
# - pmid_index.json
# - faiss_index.bin (~1GB)
# - embedding_mean.npy
# - metadata.db
```

**2. Test a search:**
```bash
python3 -c "
from src.search_engine import SearchEngine
engine = SearchEngine()
results = engine.search('achilles tendon rupture treatment')
print(f'Found {len(results)} results')
for i, result in enumerate(results[:3]):
    print(f'{i+1}. {result[\"title\"]} (score: {result[\"similarity\"]:.3f})')
"
```

**3. Start the web app:**
```bash
python3 start_app_simple.py
# Then open: http://127.0.0.1:5001
```

---

## 🐛 Common Issues & Solutions

### Issue: "Database not found"
**Solution:** Copy your `metadata.db` to the `data/` directory first

### Issue: "Model loading hangs"
**Solution:**
```bash
# Try forcing CPU mode instead of MPS/GPU
export PYTORCH_ENABLE_MPS_FALLBACK=1
python3 quick_rebuild.py
```

### Issue: "Out of memory"
**Solution:** Reduce batch size in `config/config.py`:
```python
EMBEDDING_BATCH_SIZE = 16  # Change from 32 to 16
```

### Issue: "Process stuck at X%"
**Solution:**
```bash
# Check logs
tail -50 logs/embedding_generation.log

# If truly stuck, kill and restart
pkill -9 -f embedding
python3 quick_rebuild.py
```

---

## 📊 What Gets Created

After successful rebuild:

| File | Size | Description |
|------|------|-------------|
| `data/embeddings.npy` | ~970MB | 768-dim vectors for all articles |
| `data/pmid_index.json` | ~10MB | Maps vector position to PMID |
| `data/faiss_index.bin` | ~970MB | FAISS index for fast search |
| `data/embedding_mean.npy` | ~3KB | Mean vector for query centering |
| `data/metadata.db` | ~700MB | Article metadata (titles, abstracts) |

---

## 🎯 Summary

**For a complete fresh start, just run:**

```bash
python3 quick_rebuild.py
```

**That's it!** ✨

The script handles everything automatically:
- Kills stuck processes
- Cleans up old files
- Regenerates embeddings
- Builds FAISS index
- Verifies everything works

---

## 📞 Still Having Issues?

If problems persist:

1. Check the logs: `logs/embedding_generation.log`
2. Verify database integrity: `sqlite3 data/metadata.db "SELECT COUNT(*) FROM articles;"`
3. Check available disk space: `df -h .`
4. Check available RAM: `free -h`
5. Try the manual step-by-step approach (Option 3 above)

---

Good luck! 🚀
