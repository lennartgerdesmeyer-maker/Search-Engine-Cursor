# Semantic Search Engine for PubMed Articles

A semantic search engine for PubMed articles focused on foot & ankle surgery research, using PubMedBERT embeddings and FAISS for fast similarity search.

## Features

- **Semantic Search**: Find relevant papers using natural language queries
- **PubMedBERT Embeddings**: Uses `pritamdeka/S-PubMedBert-MS-MARCO` for high-quality biomedical embeddings
- **FAISS Index**: Fast similarity search using Facebook AI Similarity Search
- **Web Interface**: Flask-based web application for easy searching
- **Citation Export**: Export results in CSV or RIS format

## Setup

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Edit `config/config.py` to set your NCBI API key:

```python
NCBI_EMAIL = "your-email@example.com"
NCBI_API_KEY = "your-api-key"
```

### 3. Download Articles

```bash
python3 src/pubmed_downloader.py
```

### 4. Generate Embeddings

```bash
python3 build_complete_system.py
```

This will:
- Generate embeddings for all articles
- Calculate embedding mean vector
- Build FAISS index

### 5. Check Progress

```bash
python3 check_progress.py
```

### 6. Run Application

```bash
python3 app.py
```

Then open http://127.0.0.1:5001 in your browser.

## Project Structure

```
.
├── app.py                      # Flask web application
├── config/                     # Configuration files
│   └── config.py              # Main configuration
├── src/                        # Source code
│   ├── embedding_generator.py  # Generate embeddings
│   ├── pubmed_downloader.py    # Download from PubMed
│   ├── search_engine.py        # Main search engine
│   └── vector_index.py         # FAISS index management
├── templates/                  # HTML templates
├── static/                     # Static assets
├── data/                       # Data files (not in git)
└── logs/                       # Log files (not in git)
```

## Usage

### Search via Web Interface

1. Start the application: `python3 app.py`
2. Open http://127.0.0.1:5001
3. Enter your search query
4. Browse and export results

### Programmatic Search

```python
from src.search_engine import SemanticSearchEngine

engine = SemanticSearchEngine()
results = engine.search("Achilles tendon rupture treatment", top_k=10)

for result in results:
    print(f"{result['title']} - {result['similarity']:.2%}")
```

## Configuration

Key settings in `config/config.py`:

- `SIMILARITY_THRESHOLD`: Minimum similarity score (default: 0.05)
- `EMBEDDING_MODEL`: Model name (default: pritamdeka/S-PubMedBert-MS-MARCO)
- `DEFAULT_TOP_K`: Number of results (default: 1000)

## Notes

- Large data files (embeddings, FAISS index, database) are excluded from git
- Embedding generation takes ~8-10 hours for ~315k papers
- Requires significant disk space for embeddings (~2GB)

## License

[Your License Here]

