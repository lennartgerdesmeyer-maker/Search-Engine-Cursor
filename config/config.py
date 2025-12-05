"""

Configuration settings for Semantic Search Engine

"""

import os

from pathlib import Path

# =============================================================================

# PATHS

# =============================================================================

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"

EXPORT_DIR = BASE_DIR / "exports"

LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist

DATA_DIR.mkdir(exist_ok=True)

EXPORT_DIR.mkdir(exist_ok=True)

LOG_DIR.mkdir(exist_ok=True)

# =============================================================================

# PUBMED / NCBI SETTINGS

# =============================================================================

# IMPORTANT: Replace with your actual email and API key

NCBI_EMAIL = "lennart.gerdesmeyer@gmail.com"

NCBI_API_KEY = "89e8755ebf109fcaad873b1d1db75ba97f09"

# =============================================================================

# MESH TERMS FOR FOOT & ANKLE SURGERY (COMPREHENSIVE)

# =============================================================================

MESH_SEARCH_QUERY = """

(

    "Foot"[MeSH Terms] OR 

    "Ankle"[MeSH Terms] OR 

    "Foot Diseases"[MeSH Terms] OR 

    "Foot Deformities"[MeSH Terms] OR 

    "Foot Deformities, Acquired"[MeSH Terms] OR

    "Foot Deformities, Congenital"[MeSH Terms] OR

    "Foot Injuries"[MeSH Terms] OR 

    "Ankle Injuries"[MeSH Terms] OR

    "Ankle Fractures"[MeSH Terms] OR

    "Tarsal Bones"[MeSH Terms] OR

    "Metatarsal Bones"[MeSH Terms] OR

    "Toe Phalanges"[MeSH Terms] OR

    "Toes"[MeSH Terms] OR

    "Hallux"[MeSH Terms] OR

    "Hallux Valgus"[MeSH Terms] OR

    "Hallux Rigidus"[MeSH Terms] OR

    "Hallux Varus"[MeSH Terms] OR

    "Hammer Toe Syndrome"[MeSH Terms] OR

    "Metatarsalgia"[MeSH Terms] OR

    "Morton Neuroma"[MeSH Terms] OR

    "Plantar Fasciitis"[MeSH Terms] OR

    "Flatfoot"[MeSH Terms] OR

    "Pes Cavus"[MeSH Terms] OR

    "Talipes"[MeSH Terms] OR

    "Clubfoot"[MeSH Terms] OR

    "Talipes Cavus"[MeSH Terms] OR

    "Talipes Equinovarus"[MeSH Terms] OR

    "Achilles Tendon"[MeSH Terms] OR

    "Tendinopathy"[MeSH Terms] OR

    "Calcaneus"[MeSH Terms] OR

    "Talus"[MeSH Terms] OR

    "Subtalar Joint"[MeSH Terms] OR

    "Tarsal Joints"[MeSH Terms] OR

    "Metatarsophalangeal Joint"[MeSH Terms] OR

    "Tarsal Tunnel Syndrome"[MeSH Terms] OR

    "Diabetic Foot"[MeSH Terms] OR

    "Charcot-Marie-Tooth Disease"[MeSH Terms] OR

    "Posterior Tibial Tendon Dysfunction"[MeSH Terms] OR

    "Osteochondritis Dissecans"[MeSH Terms] OR

    "Lisfranc Injuries"[MeSH Terms] OR

    "Sesamoid Bones"[MeSH Terms] OR

    "Bunion"[MeSH Terms] OR

    "Bunionette"[MeSH Terms] OR

    "Freiberg Disease"[MeSH Terms] OR

    "Kohler Disease"[MeSH Terms] OR

    "Sever Disease"[MeSH Terms] OR

    "Haglund Deformity"[MeSH Terms] OR

    

    "foot"[Title/Abstract] OR 

    "ankle"[Title/Abstract] OR

    "hindfoot"[Title/Abstract] OR

    "midfoot"[Title/Abstract] OR

    "forefoot"[Title/Abstract] OR

    "rearfoot"[Title/Abstract] OR

    "hallux"[Title/Abstract] OR

    "metatarsal"[Title/Abstract] OR

    "metatarsals"[Title/Abstract] OR

    "phalanges"[Title/Abstract] OR

    "tarsal"[Title/Abstract] OR

    "tarsals"[Title/Abstract] OR

    "calcaneus"[Title/Abstract] OR

    "calcaneal"[Title/Abstract] OR

    "talus"[Title/Abstract] OR

    "talar"[Title/Abstract] OR

    "navicular"[Title/Abstract] OR

    "cuboid"[Title/Abstract] OR

    "cuneiform"[Title/Abstract] OR

    "achilles"[Title/Abstract] OR

    "plantar"[Title/Abstract] OR

    "dorsiflexion"[Title/Abstract] OR

    "plantarflexion"[Title/Abstract] OR

    "subtalar"[Title/Abstract] OR

    "talonavicular"[Title/Abstract] OR

    "calcaneocuboid"[Title/Abstract] OR

    "lisfranc"[Title/Abstract] OR

    "chopart"[Title/Abstract] OR

    "bunion"[Title/Abstract] OR

    "bunionette"[Title/Abstract] OR

    "hammertoe"[Title/Abstract] OR

    "hammer toe"[Title/Abstract] OR

    "claw toe"[Title/Abstract] OR

    "mallet toe"[Title/Abstract] OR

    "flatfoot"[Title/Abstract] OR

    "flat foot"[Title/Abstract] OR

    "pes planus"[Title/Abstract] OR

    "pes cavus"[Title/Abstract] OR

    "cavovarus"[Title/Abstract] OR

    "equinus"[Title/Abstract] OR

    "clubfoot"[Title/Abstract] OR

    "club foot"[Title/Abstract] OR

    "talipes"[Title/Abstract] OR

    "syndesmosis"[Title/Abstract] OR

    "syndesmotic"[Title/Abstract] OR

    "morton neuroma"[Title/Abstract] OR

    "interdigital neuroma"[Title/Abstract] OR

    "sesamoid"[Title/Abstract] OR

    "sesamoiditis"[Title/Abstract]

)

"""

# =============================================================================

# EMBEDDING MODEL SETTINGS

# =============================================================================

EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

EMBEDDING_DIMENSION = 768

# =============================================================================

# SEARCH SETTINGS

# =============================================================================

DEFAULT_TOP_K = 1000

MAX_TOP_K = 10000  # Increased from 5000 to allow more results for better recall

SIMILARITY_THRESHOLD = 0.10  # Balanced threshold: 0.05 was too permissive, 0.10 filters weak matches

# Recommended thresholds for different use cases:
# - High precision: 0.20-0.30 (fewer results, more relevant)
# - Balanced: 0.10-0.15 (default)
# - High recall: 0.05 (more results, some irrelevant)

# Validation mode settings (for maximum recall when validating against known studies)
VALIDATION_MODE = {
    "min_similarity": 0.0,      # No threshold for maximum recall
    "top_k": 5000,              # Maximum results
    "use_reranker": False,      # Reranker may reduce recall
    "faiss_candidates": 10000,  # Request maximum candidates
}

# =============================================================================

# DATABASE SETTINGS

# =============================================================================

VECTOR_DB_PATH = DATA_DIR / "vector_store"

METADATA_DB_PATH = DATA_DIR / "metadata.db"

ARTICLES_PARQUET_PATH = DATA_DIR / "articles.parquet"

# =============================================================================

# BATCH PROCESSING SETTINGS

# =============================================================================

PUBMED_BATCH_SIZE = 500

EMBEDDING_BATCH_SIZE = 32

SAVE_CHECKPOINT_EVERY = 5000

# =============================================================================

# WEB INTERFACE SETTINGS

# =============================================================================

FLASK_HOST = "127.0.0.1"

FLASK_PORT = 5001  # Changed from 5000 to avoid conflict with macOS AirPlay Receiver

DEBUG_MODE = True
