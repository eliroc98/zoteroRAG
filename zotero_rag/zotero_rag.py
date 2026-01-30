"""Main orchestration class for Zotero RAG system."""

import os
# Suppress noisy progress bars that can trigger BrokenPipe in Streamlit
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
from typing import List, Dict, Tuple
import warnings
import nltk

from models import Paragraph, Answer
from zotero_db import ZoteroDatabase
from folder_source import FolderPDFSource
from pdf_processor import PDFProcessor
from indexer import Indexer
from reranker import Reranker
from qa_engine import QAEngine
from highlighter import PDFHighlighter

warnings.filterwarnings('ignore', message='.*position_ids.*')

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler('zotero_rag.log', mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers only if they haven't been added yet
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class ZoteroRAG:
    """Main orchestration class for the Zotero RAG pipeline."""
    
    def __init__(self, 
                 zotero_data_dir: str = None, 
                 collection_name: str = None,
                 source_type: str = 'zotero',
                 folder_path: str = None,
                 model_name: str = "BAAI/bge-base-en-v1.5", 
                 qa_model: str = "deepset/roberta-base-squad2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 grobid_url: str = "http://localhost:8070", 
                 grobid_timeout: int = 180,
                 model_device: str = None, 
                 encode_batch_size: int = None,
                 rerank_batch_size: int = None,
                 tei_cache_dir: str = None,
                 output_base_dir: str = "output"):
        """Initialize the RAG system.
        
        Args:
            zotero_data_dir: Path to Zotero data directory. Auto-detect if None (for Zotero mode).
            collection_name: Name of Zotero collection to use. If None, use entire library (for Zotero mode).
            source_type: Type of PDF source - 'zotero' or 'folder'.
            folder_path: Path to folder containing PDFs (for folder mode).
            model_name: Name of the sentence transformer model for embeddings.
            qa_model: Name of the QA model for answer extraction.
            reranker_model: Name of the cross-encoder model for reranking.
            grobid_url: URL of the GROBID service.
            grobid_timeout: Timeout in seconds for GROBID requests.
            model_device: Device to use for models ('cpu', 'cuda', 'mps'). Auto-detect if None.
            encode_batch_size: Batch size for encoding. If None, auto-detect (targets 75% memory).
            rerank_batch_size: Batch size for reranking. If None, auto-detect (targets 75% memory).
            tei_cache_dir: Directory to cache TEI XML outputs.
            output_base_dir: Base directory for storing indexes and outputs.
        """
        self.source_type = source_type
        self.collection_name = collection_name
        self.folder_path = folder_path
        self.output_base_dir = output_base_dir
        
        # Initialize the appropriate source
        if source_type == 'folder':
            if not folder_path:
                raise ValueError("folder_path is required when source_type='folder'")
            self.source = FolderPDFSource(folder_path)
            # Use folder name for cache directory
            source_name = os.path.basename(folder_path)
        else:
            self.source = ZoteroDatabase(zotero_data_dir)
            source_name = collection_name
        
        # Set up TEI cache directory
        base_cache = tei_cache_dir or os.path.join(output_base_dir, "tei_cache")
        source_folder = self._sanitize_filename(source_name)
        pdf_cache_dir = os.path.join(base_cache, source_folder)
        
        self.pdf_processor = PDFProcessor(
            grobid_url=grobid_url,
            grobid_timeout=grobid_timeout,
            tei_cache_dir=pdf_cache_dir
        )
        
        self.indexer = Indexer(
            model_name=model_name,
            device=model_device,
            encode_batch_size=encode_batch_size
        )
        
        self.reranker = Reranker(
            model_name=reranker_model,
            device=model_device,
            batch_size=rerank_batch_size
        )
        
        self.qa_engine = QAEngine(
            model_name=qa_model,
            device=model_device
        )
        
        self.highlighter = PDFHighlighter()
        
        # Set index paths based on source
        self.indexer.set_index_paths(source_name, output_base_dir)
        
        # Color management for multi-query highlighting
        self.query_colors = [
            (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.5, 0.0),
            (0.5, 1.0, 0.5), (1.0, 0.7, 0.8), (0.7, 0.5, 1.0),
        ]
        self.query_color_map = {}
        
        # For debugging/inspection
        self.last_candidates = []
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Converts a string into a safe folder/file name."""
        import re
        if not name:
            return "_All_Library"
        s = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        s = re.sub(r'(?u)[^-\w.]', '', s)
        return s
    
    @staticmethod
    def list_collections(zotero_data_dir: str = None) -> List[Dict]:
        """Load collections from the Zotero database.
        
        Args:
            zotero_data_dir: Path to Zotero data directory. Auto-detect if None.
            
        Returns:
            List of dictionaries with 'id', 'name', and 'parent_id' keys.
        """
        db = ZoteroDatabase(zotero_data_dir)
        return db.list_collections()
    
    @property
    def paragraphs(self):
        """Access paragraphs from the indexer (backward compatibility)."""
        return self.indexer.paragraphs
    
    @property
    def index(self):
        """Access FAISS index from the indexer (backward compatibility)."""
        return self.indexer.index
    
    @property
    def index_path(self):
        """Access index path from the indexer (backward compatibility)."""
        return self.indexer.index_path
    
    @property
    def chunks_path(self):
        """Access chunks path from the indexer (backward compatibility)."""
        return self.indexer.chunks_path
    
    def get_query_color(self, query: str) -> Tuple[float, float, float]:
        """Get a consistent color for a query string.
        
        Args:
            query: Query string.
            
        Returns:
            RGB color tuple.
        """
        if query not in self.query_color_map:
            color_idx = len(self.query_color_map) % len(self.query_colors)
            self.query_color_map[query] = self.query_colors[color_idx]
        return self.query_color_map[query]
    
    def set_index_paths(self, base_filename: str = None):
        """Set the file paths for the index and chunks.
        
        Args:
            base_filename: Optional base path/name. If full path provided, uses it.
                          If just a name, creates in collection directory.
                          If None, auto-generates from collection and model.
        """
        if base_filename and os.path.dirname(base_filename):
            # Full path provided - extract collection folder and filename
            self.indexer.index_path = f"{base_filename}.index"
            self.indexer.chunks_path = f"{base_filename}.pkl"
            logger.info(f"Index paths set to: {self.indexer.index_path} and {self.indexer.chunks_path}")
        else:
            # Let indexer handle it
            self.indexer.set_index_paths(self.collection_name, self.output_base_dir)
    
    def build_index(self, force_rebuild: bool = False, progress_callback=None) -> int:
        """Build or load the FAISS index from Zotero PDFs.
        
        Args:
            force_rebuild: If True, rebuild even if index exists.
            progress_callback: Function(stage, current, total, message) for progress updates.
                             stage is 'pdf' or 'encoding'.
                             
        Returns:
            Number of paragraphs indexed.
        """
        # Check if we can load existing index
        if not force_rebuild and self.indexer.index_exists():
            if progress_callback:
                progress_callback('pdf', 1, 1, "Loading existing index...")
            return self.indexer.load_index()
        
        # Get items from source (Zotero or folder)
        items = self.source.get_items(self.collection_name)
        if not items:
            source_desc = f"folder {self.folder_path}" if self.source_type == 'folder' else "Zotero collection/library"
            raise ValueError(f"No PDF items found in the specified {source_desc}.")
        
        # Stage 1: Process PDFs and extract paragraphs
        all_paragraphs = []
        for idx, item in enumerate(items):
            if progress_callback:
                progress_callback('pdf', idx, len(items), 
                                f"Processing: {item['title'][:50]}...")
            
            paragraph_tuples = self.pdf_processor.extract_text_chunks(
                item['path'], 
                item['title']
            )
            
            for text, page_num, section, sentences in paragraph_tuples:
                # Filter by section type if needed
                if not self.pdf_processor.CONTENT_SECTIONS.get(section, True):
                    continue
                    
                sentence_count = len(sentences)
                paragraph = Paragraph(
                    text=text,
                    pdf_path=item['path'],
                    page_num=page_num,
                    item_key=item['key'],
                    title=item['title'],
                    section=section,
                    sentence_count=sentence_count,
                    sentences=sentences
                )
                all_paragraphs.append(paragraph)
        
        if not all_paragraphs:
            raise ValueError("No text could be extracted from the PDFs.")
        
        # Stage 2: Build index
        return self.indexer.build_index(
            all_paragraphs, 
            force_rebuild=force_rebuild,
            progress_callback=progress_callback
        )
    
    def index_exists(self) -> bool:
        """Check if the index exists for the current collection."""
        return self.indexer.index_exists()
    
    def load_index(self) -> int:
        """Load an existing FAISS index from disk.
        
        Returns:
            Number of paragraphs loaded.
        """
        return self.indexer.load_index()
    
    def answer_question(self, 
                       question: str, 
                       retrieval_threshold: float = 2.0, 
                       qa_score_threshold: float = 0.0, 
                       rerank_threshold: float = 0.25, 
                       progress_callback=None, 
                       rerank_callback=None,
                       question_type: str = 'general',
                       custom_config: dict = None,
                       num_paraphrases: int = 2,
                       highlight_color: Tuple[float, float, float] = None,
                       question_variations: List[str] = None) -> List[Answer]:
        """Answer a question using the full RAG pipeline.
        
        Pipeline stages:
        1. FAISS Retrieval (Range Search)
        2. CrossEncoder Reranking (Threshold Filtering)
        3. QA Extraction (with Context Overlap/Sliding Window)
        
        Args:
            question: The question to answer.
            retrieval_threshold: L2 distance threshold for initial retrieval.
            qa_score_threshold: Minimum QA confidence score to keep answers.
            rerank_threshold: Minimum rerank probability to keep candidates.
            progress_callback: Function(current, total, message) for QA progress.
            rerank_callback: Function(current, total, message) for rerank progress.
            question_type: Type of question (factoid, explanation, methodology, etc.).
            custom_config: Custom configuration dict to override preset config.
            num_paraphrases: Number of question paraphrases to generate (0 = disabled).
            highlight_color: RGB tuple (0-1) for highlighting. If None, use query-based color.
            question_variations: Pre-generated question variations to use. If None, generate them.
            
        Returns:
            List of Answer objects, deduplicated and sorted by score.
        """
        if not self.indexer.index:
            raise ValueError("Index is not built. Call build_index() first.")
        
        # Stage 0: Expand question if enabled and variations not provided
        if question_variations is None:
            question_variations = [question]  # Always include original
            if self.qa_engine.enable_question_expansion and num_paraphrases > 0:
                question_variations = self.qa_engine.expand_question(question, num_variations=num_paraphrases)
                logger.info(f"Question expansion: {len(question_variations)} variations generated")
            elif num_paraphrases == 0:
                logger.info("Question paraphrasing disabled by user")
        else:
            logger.info(f"Using {len(question_variations)} pre-selected question variations")
        
        # Stage 1: Retrieve candidate paragraphs (FAISS)
        # Search with all question variations and merge results
        all_candidates = []
        seen_paragraphs = set()
        
        for i, q_var in enumerate(question_variations):
            var_candidates = self.indexer.search(q_var, retrieval_threshold)
            logger.debug(f"Variation {i}: '{q_var}' -> {len(var_candidates)} candidates")
            
            # Add unseen candidates
            for para, score, idx in var_candidates:
                para_id = (para.pdf_path, para.page_num, para.text[:100])  # Unique identifier
                if para_id not in seen_paragraphs:
                    seen_paragraphs.add(para_id)
                    all_candidates.append((para, score, idx))
        
        # Sort by retrieval score
        all_candidates.sort(key=lambda x: x[1])
        candidates = all_candidates
        
        logger.debug(f"Question: {question}")
        logger.debug(f"Retrieved {len(candidates)} unique paragraphs from {len(question_variations)} variations")
        
        if not candidates:
            self.last_candidates = []
            return []
        
        # Store for debugging
        self.last_candidates = [
            {
                'paragraph': c[0],
                'retrieval_score': c[1],
                'kept': True
            }
            for c in candidates
        ]
        
        # Stage 2: Rerank and Filter (CrossEncoder)
        reranked = self.reranker.rerank(
            question, 
            candidates, 
            rerank_threshold,
            progress_callback=rerank_callback,
            query_variations=question_variations
        )
        
        # Update debug info with rerank results
        reranked_texts = {c[0].text for c in reranked}
        for c in self.last_candidates:
            c['kept'] = c['paragraph'].text in reranked_texts
        
        if not reranked:
            return []
        
        # Stage 3: Extract answers (QA Model)
        # Use provided color or get a query-based color
        if highlight_color is None:
            color = self.get_query_color(question)
        else:
            color = highlight_color
        
        answers = self.qa_engine.extract_answers(
            question,
            reranked,
            self.indexer.paragraphs,
            qa_score_threshold=qa_score_threshold,
            color=color,
            progress_callback=progress_callback,
            question_variations=question_variations,
            question_type=question_type,
            custom_config=custom_config
        )
        
        return answers
    
    def highlight_pdf(self, answers_for_pdf: List[Answer], output_path: str) -> str:
        """Highlight PDF using TEI sentence coordinates.
        
        Args:
            answers_for_pdf: List of Answer objects from the same PDF.
            output_path: Path where the highlighted PDF should be saved.
            
        Returns:
            Path to the highlighted PDF, or None if highlighting failed.
        """
        return self.highlighter.highlight_pdf(answers_for_pdf, output_path)
