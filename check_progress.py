#!/usr/bin/env python3
"""
Simple progress checker - just run: python3 check_progress.py
"""
import json
from pathlib import Path
from datetime import datetime
from dateutil import parser

checkpoint_path = Path("data/embedding_checkpoint.json")
embeddings_path = Path("data/embeddings.npy")
mean_path = Path("data/embedding_mean.npy")
faiss_path = Path("data/faiss_index.bin")

print("=" * 70)
print("📊 EMBEDDING GENERATION PROGRESS")
print("=" * 70)
print()

if checkpoint_path.exists():
    with open(checkpoint_path) as f:
        cp = json.load(f)
    
    processed = cp['processed']
    total = cp['total']
    pct = (processed / total * 100) if total > 0 else 0
    remaining = total - processed
    
    # Progress bar
    width = 50
    filled = int(width * pct / 100)
    bar = '█' * filled + '░' * (width - filled)
    
    print(f"Progress: {processed:,} / {total:,} papers ({pct:.2f}%)")
    print(f"[{bar}]")
    print(f"Remaining: {remaining:,} papers")
    print()
    
    if remaining > 0:
        # Estimate: ~300 papers per minute
        est_min = remaining / 300
        est_hours = est_min / 60
        print(f"⏱️  Estimated time: ~{est_hours:.1f} hours ({est_min:.0f} minutes)")
        print()
    
    # Last update
    try:
        cp_time = parser.parse(cp['timestamp'])
        now = datetime.now(cp_time.tzinfo)
        age_min = (now - cp_time).total_seconds() / 60
        if age_min < 1:
            print(f"🕐 Last update: Just now")
        elif age_min < 60:
            print(f"🕐 Last update: {int(age_min)} minutes ago")
        else:
            print(f"🕐 Last update: {age_min/60:.1f} hours ago")
    except:
        print(f"🕐 Last checkpoint: {cp.get('timestamp', 'unknown')[:19]}")
else:
    print("⏳ Embedding generation hasn't started yet")
    print("   (checkpoint file not found)")
    print()

# File sizes
if embeddings_path.exists():
    sz_mb = embeddings_path.stat().st_size / 1024 / 1024
    print(f"💾 Embeddings file: {sz_mb:.1f} MB")
if mean_path.exists():
    print(f"✅ Embedding mean: Calculated")
if faiss_path.exists():
    sz_mb = faiss_path.stat().st_size / 1024 / 1024
    print(f"✅ FAISS index: {sz_mb:.1f} MB")

print()
print("=" * 70)

