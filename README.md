# 📚 Zotero RAG Navigator

A question-answering system for your local Zotero library or PDF folders featuring a multi-stage RAG pipeline with GROBID parsing, semantic search, reranking, and extractive QA. Generate precise answers from your research papers with automatic highlighting and question expansion.

## Features

**3-Stage RAG Pipeline**: FAISS retrieval → CrossEncoder reranking → Extractive QA

- **Flexible PDF sources**: Zotero collections or any folder of PDFs
- GROBID sentence-level parsing with coordinates for precise PDF highlighting
- Question expansion via automatic paraphrasing for improved recall
- Question type presets (factoid, methodology, explanation, comparison)
- Sliding window QA for answers spanning paragraph boundaries
- Multi-color highlighting for multiple queries
- Streamlit web interface with real-time progress tracking

## Requirements

- Python 3.11+
- Zotero (with local PDF storage) **OR** a folder containing PDF files
- GROBID server (Docker)
- PyTorch with MPS/CUDA support (optional, CPU works too)

## Project Structure

```
.
├── zotero_rag/              # Main package
│   ├── app.py               # Streamlit web interface
│   ├── run_from_config.py   # Programmatic YAML config runner
│   ├── zotero_rag.py        # Main orchestration class
│   ├── zotero_db.py         # Zotero SQLite database interface
│   ├── folder_source.py     # Folder-based PDF source
│   ├── pdf_processor.py     # GROBID client and TEI parsing
│   ├── indexer.py           # FAISS indexing and retrieval
│   ├── reranker.py          # CrossEncoder reranking
│   ├── qa_engine.py         # Extractive QA with question expansion
│   ├── highlighter.py       # PDF annotation using coordinates
│   └── models.py            # Data classes (Paragraph, Answer)
│
├── example_configs/              # Example YAML configurations
│   ├── example_config.yaml       # Full YAML config with documentation
│   ├── advanced_config.yaml      # Advanced YAML config with custom paraphrases
│   ├── folder_example_config.yaml # Example config for folder-based PDFs
│   ├── question_type_presets.yaml # Question type preset documentation
│   └── highlighter_colors.html   # Color reference for highlighting
│
├── pyproject.toml           # Poetry dependencies
├── README.md                # This file
└── LICENSE                  # GPL v3.0 license

output/                     # Output directory (indexes, cache, highlighted PDFs)
├── {collection}/
│   ├── index_{model}.index  # FAISS index
│   ├── index_{model}.pkl    # Paragraph metadata
│   └── highlighted/         # Highlighted PDFs
└── tei_cache/
    └── {collection}/
        └── {hash}.tei.xml   # GROBID output cache
```

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
poetry run streamlit run zotero_rag/app.py
```

Then navigate to `http://localhost:8501`

## Usage

### Initial Setup

1. **Output Directory** (first page)
   - Specify where to store indexes, TEI cache, and highlighted PDFs
   - Default: `./literature_output`

2. **Select PDF Source**
   - **📚 Zotero Collection**: Use PDFs from your Zotero library
   - **📁 Folder of PDFs**: Use PDFs from any folder on your system
   
   **For Zotero**: Choose a collection or "All Library"
   
   **For Folder**: Enter the full path to a folder containing PDFs (will search recursively)
   - Example: `/Users/username/Documents/research_papers`
   - All PDFs in the folder and subfolders will be indexed

3. **GROBID Configuration**
   - Service URL (default: `http://localhost:8070`)
   - Leave default if running locally
   - Required for new PDFs; cached TEI files are reused

4. **Select Embedding Model**
   - Model name: Any HuggingFace SentenceTransformer (e.g., `BAAI/bge-base-en-v1.5`)
   - Device: auto/cpu/mps/cuda
   - Batch sizes: Auto-detected or manual override
   - Click "Load Model"

5. **Build Index**
   - First time: Click "Build Index" to process all PDFs
   - System auto-detects safe batch size
   - Progress shown per batch
   - Subsequent loads use cached indexes

### Question Answering

1. **Enter a Question**
   - Natural language queries supported
   - Example: "What methods are used for circuit interpretability?"

2. **Question Type Selection** (Optional)
   - **Factoid**: Short, precise answers (e.g., "Who invented transformers?")
   - **Methodology**: Detailed process explanations (e.g., "How does attention mechanism work?")
   - **Explanation**: Conceptual understanding (e.g., "Why are transformers effective?")
   - **Comparison**: Contrasting approaches (e.g., "What's the difference between BERT and GPT?")
   - **General**: Balanced settings for mixed queries

3. **Question Expansion** (Optional)
   - Generate paraphrases to improve retrieval
   - Select/edit which variations to use
   - Automatically merges results from all variations

4. **Adjust Parameters** (Advanced)
   - **Retrieval threshold**: L2 distance (lower = more candidates)
   - **Rerank threshold**: CrossEncoder score (0-1, higher = stricter)
   - **QA threshold**: Answer confidence (0-1, higher = more confident)
   - **Answer length**: Max characters per answer
   - **Min words**: Minimum answer length filter

5. **Highlight Color**
   - Choose from presets (Yellow, Cyan, Orange, Green, Pink, Purple)
   - Or use custom RGB color picker

6. **Search & Navigate**
   - View answers with context and scores
   - Navigate results with Previous/Next buttons
   - See PDF source, page number, and section
   - Click "Open PDF" to view in default viewer
   - Click "Highlight PDF" to add colored annotations

7. **Multi-Query Highlighting**
   - Run multiple queries with different colors
   - All highlights accumulate in the same PDF
   - Perfect for exploring different aspects of a paper

## Architecture

### Data Flow

```
PDF Source (Zotero Library OR Folder)
    ↓
PDF Selection
    ↓
GROBID Processing (sentence segmentation + coordinates)
    ↓
TEI Cache (mtime-keyed, persistent)
    ↓
Paragraph Extraction (section classification)
    ↓
SentenceTransformer Encoding (auto batch-size)
    ↓
FAISS IndexFlatL2
```

### Query Pipeline

```
User Question
    ↓
Question Expansion (optional paraphrasing)
    ↓
FAISS Retrieval (L2 range search, all variations)
    ↓
CrossEncoder Reranking (adaptive threshold)
    ↓
Extractive QA (sliding window with context overlap)
    ↓
Answer Deduplication & Scoring
    ↓
PDF Highlighting (TEI coordinate mapping)
```

## Configuration

### Environment Variables (Optional)

```bash
# Default Zotero directory (auto-detected from ~/Zotero, ~/Documents/Zotero, ~/.zotero)
export ZOTERO_DATA_DIR=/path/to/zotero

# GROBID timeout (seconds)
export GROBID_TIMEOUT=180

# Disable progress bars (useful in headless environments)
export TQDM_DISABLE=1

# Tokenizer parallelism
export TOKENIZERS_PARALLELISM=false
```

### Question Type Presets

Each question type has optimized parameters:

- **Factoid**: Stricter QA threshold, shorter answers, entity preference
- **Methodology**: Very lenient threshold, longer answers, section diversity
- **Explanation**: Lenient threshold, medium-length answers
- **Comparison**: Moderate threshold, diverse sources preferred
- **General**: Balanced settings for mixed queries

### Programmatic Usage

#### Option 1: YAML Configuration (Recommended)

Create a YAML config file (see [example_configs/example_config.yaml](example_configs/example_config.yaml) for full documentation):

```yaml
# config.yaml
collection_name: "My Research Papers"
output_base_dir: ./output

# Model settings (optional - uses defaults if not specified)
model_name: BAAI/bge-base-en-v1.5
qa_model: deepset/roberta-base-squad2
reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
model_device: null  # null for auto-detect, or 'cpu', 'cuda', 'mps'

# Batch sizes (null for auto-detect)
encode_batch_size: null
rerank_batch_size: null

# Questions to answer
questions:
  - question: "What are transformers?"
    highlight_color: [1, 1, 0]  # Yellow
  
  - question: "How does attention work?"
    paraphrases:
      - "Explain the attention mechanism"
      - "What is the attention layer?"
    highlight_color: [0, 1, 0]  # Green
    question_type: explanation
    
  - question: "What datasets were used?"
    retrieval_threshold: 0.5
    rerank_threshold: 0.3
    highlight_color: [0, 0.5, 1]  # Blue
    question_type: factoid
    custom_config:
      qa_score_threshold: 0.3
      max_answer_length: 100
      prefer_entities: true

# Default settings for all questions
defaults:
  num_paraphrases: 2
  retrieval_threshold: 0.7
  rerank_threshold: 0.25
  highlight_color: [1, 1, 0]
  question_type: general

# Output settings
create_highlighted_pdfs: true
output_results_file: results.json
```

Run from command line:
```bash
poetry run python -m zotero_rag.run_from_config config.yaml
```

Or use programmatically:
```python
from zotero_rag.run_from_config import run_from_config

# Run config and get results
results = run_from_config('config.yaml')

# Results is a dict: {question: [Answer, Answer, ...]}
for question, answers in results.items():
    print(f"\nQuestion: {question}")
    for answer in answers[:3]:
        print(f"  - {answer.text} (score: {answer.score:.3f})")
        print(f"    PDF: {answer.pdf_path}, Page: {answer.page_num}")
```

#### Option 2: Python API

```python
from zotero_rag import ZoteroRAG

# Initialize with full configuration
rag = ZoteroRAG(
    collection_name="circuits",
    model_name="BAAI/bge-base-en-v1.5",
    qa_model="deepset/roberta-base-squad2",
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    grobid_url="http://localhost:8070",
    model_device="mps",  # or "cpu", "cuda", None for auto
    encode_batch_size=32,  # or None for auto-detection
    rerank_batch_size=32,
    output_base_dir="./literature_output"
)

# Set index paths
rag.set_index_paths()

# Build index (or load if exists)
num_chunks = rag.build_index(force_rebuild=False)
print(f"Indexed {num_chunks} paragraphs")

# Ask a question with full pipeline
answers = rag.answer_question(
    question="How does attention mechanism work?",
    retrieval_threshold=0.7,     # L2 distance for FAISS
    rerank_threshold=0.25,       # CrossEncoder score (0-1)
    question_type='methodology', # Use preset optimization
    num_paraphrases=2,           # Generate question variations
    highlight_color=(1.0, 1.0, 0.0)  # Yellow highlights (RGB 0-1)
)

# Display answers
for answer in answers:
    print(f"\n{'='*80}")
    print(f"PDF: {answer.pdf_path}")
    print(f"Page: {answer.page_num} | Section: {answer.section}")
    print(f"Score: {answer.score:.3f} | Rerank: {answer.rerank_score:.3f}")
    print(f"\nContext: {answer.context}")
    print(f"Answer: {answer.text}")

# Group answers by PDF and highlight
from collections import defaultdict
by_pdf = defaultdict(list)
for answer in answers:
    by_pdf[answer.pdf_path].append(answer)

for pdf_path, pdf_answers in by_pdf.items():
    output_path = f"./highlighted_{pdf_path.split('/')[-1]}"
    rag.highlight_pdf(pdf_answers, output_path)
    print(f"Highlighted: {output_path}")

# Advanced: Pre-defined question variations (skip auto-generation)
predefined_paraphrases = [
    "How does attention mechanism work?",
    "Explain the attention layer",
    "What is the attention mechanism?"
]

answers = rag.answer_question(
    question="How does attention mechanism work?",
    question_variations=predefined_paraphrases,
    highlight_color=(0, 1, 0)  # Green
)

# Advanced: Custom configuration (overrides question_type presets)
custom_config = {
    'qa_score_threshold': 0.1,    # Minimum QA confidence (0-1)
    'max_answer_length': 200,      # Max characters per answer
    'min_answer_words': 3,         # Minimum word count filter
    'prefer_entities': False,      # Prioritize named entities
    'section_diversity': True,     # Prefer answers from different sections
    'priority_sections': ['abstract', 'methodology']  # Preferred sections
}

answers = rag.answer_question(
    question="What datasets were used?",
    question_type='factoid',       # Base preset to start from
    custom_config=custom_config    # Custom overrides
)
```

## Advanced Features

### Question Type Presets

The system includes optimized presets for different query types (see [example_configs/question_type_presets.yaml](example_configs/question_type_presets.yaml)):

- **factoid**: Short, precise answers ("Who invented transformers?", "What learning rate was used?")
- **methodology**: Detailed process explanations ("How does attention work?", "How is the model trained?")
- **explanation**: Conceptual understanding ("Why are transformers effective?", "What makes BERT different?")
- **comparison**: Contrasting approaches ("What's the difference between BERT and GPT?")
- **definition**: Clear term explanations ("What is BERT?", "Define attention mechanism")
- **general**: Balanced settings for mixed queries

Each preset automatically adjusts retrieval thresholds, answer length, and filtering criteria.

### Multi-Color Highlighting

Run multiple queries with different colors to explore different aspects:

```python
colors = [(1, 1, 0), (0, 1, 0), (1, 0, 1)]  # Yellow, Green, Pink
questions = [
    "What are transformers?",
    "How does attention work?",
    "What datasets were used?"
]

for question, color in zip(questions, colors):
    answers = rag.answer_question(question, highlight_color=color)
    # Highlights accumulate in the same PDFs
```

See [example_configs/highlighter_colors.html](example_configs/highlighter_colors.html) for color reference.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{zoterorag2026,
  author = {Elisabetta Rocchetti},
  title = {Zotero RAG Navigator: Multi-Stage Question Answering for Research Libraries},
  year = {2026},
  url = {https://github.com/eliroc98/zoteroRAG}
}
```

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
