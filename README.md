# 📚 Zotero RAG Navigator

A semantic search system for your local Zotero library using advanced PDF parsing (GROBID), sentence-level embedding, and FAISS indexing. Query your research papers with natural language and get highlighted PDFs.

## Requirements

- Python 3.11+
- Zotero (with local PDF storage)
- GROBID server (Docker)
- PyTorch with MPS/CUDA support (optional, CPU works too)

## Installation

### 1. Clone & Setup Environment

```bash
git clone https://github.com/eliroc98/zoteroRAG.git
cd zoteroRAG
poetry install
```

### 2. Start GROBID Service

```bash
docker run -d -p 8070:8070 grobid/grobid:latest
```

Verify it's running:
```bash
curl http://localhost:8070/api/isalive
```

### 3. Run the App

```bash
poetry run streamlit run app.py
```

Then navigate to `http://localhost:8501`

## Usage

### Initial Setup

1. **Output Directory** (first page)
   - Specify where to store indexes, TEI cache, and highlighted PDFs
   - Default: `./output`

2. **GROBID Configuration**
   - Service URL (default: `http://localhost:8070`)
   - Leave default if running locally

3. **Select Collection**
   - Choose a Zotero collection or "All Library"

4. **Select Embedding Model**
   - Model name: Any HuggingFace SentenceTransformer (e.g., `BAAI/bge-small-en-v1.5`)
   - Device: auto/cpu/mps/cuda
   - Click "Load Model"

5. **Build Index**
   - First time: Click "Build Index" to process all PDFs
   - System auto-detects safe batch size
   - Progress shown per batch
   - Subsequent loads use cached indexes

### Search & Highlight

1. Enter a natural language query
2. Adjust similarity threshold (lower = more results, default 1.2)
3. Navigate results with Previous/Next buttons
4. Click "Open PDF" to view in your default PDF viewer
5. Click "Highlight All" to add colored highlights and annotations
6. Run another query to add more highlights in different colors

## Data Flow

```
Zotero PDFs
    ↓
GROBID (sentence segmentation + coords)
    ↓
TEI Cache (mtime-keyed)
    ↓
Sentence Extraction (section classification)
    ↓
SentenceTransformer Encoding (auto batch-size)
    ↓
FAISS IndexFlatL2
    ↓
Search (keyword filter + L2 similarity)
    ↓
Highlighted PDF Output
```

## Configuration

### Environment Variables (Optional)

```bash
# Default Zotero directory (auto-detected from ~/Zotero, ~/Documents/Zotero, ~/.zotero)
export ZOTERO_DATA_DIR=/path/to/zotero

# GROBID timeout (seconds)
export GROBID_TIMEOUT=180
```

### Programmatic Usage

```python
from zotero_rag import ZoteroRAG

# Initialize
rag = ZoteroRAG(
    collection_name="circuits",
    model_name="BAAI/bge-m3",
    grobid_url="http://localhost:8070",
    model_device="mps",  # or "cpu", "cuda", None for auto
    output_base_dir="./output"
)

# Set index paths
rag.set_index_paths()

# Build index
num_chunks = rag.build_index(force_rebuild=False)

# Search
results = rag.search("circuit interpretability", threshold=1.5)
for chunk, distance in results:
    print(f"{chunk.text} (page {chunk.page_num}, score: {distance:.3f})")

# Highlight
rag.highlight_pdf([chunk for chunk, _ in results], "output/highlighted.pdf")
```

## Citation

If you use this system in your research, please cite:

```bibtex
@software{zoterorag2026,
  author = {Elisabetta Rocchetti},
  title = {Zotero RAG Navigator},
  year = {2026},
  url = {https://github.com/eliroc98/zoteroRAG}
}
```

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
