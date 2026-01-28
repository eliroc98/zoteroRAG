import os
# Suppress noisy progress bars that can trigger BrokenPipe in Streamlit
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sqlite3
import re
import threading
import hashlib
import logging
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import faiss
from dataclasses import dataclass, field
import pickle
import warnings
import requests
import tempfile
import shutil
import xml.etree.ElementTree as ET
import nltk
from grobid_client.grobid_client import GrobidClient

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

@dataclass
class Paragraph:
    """Represents a paragraph-level chunk for QA."""
    text: str
    pdf_path: str
    page_num: int
    item_key: str
    title: str
    section: str = "body"  # section type: body, abstract, intro, etc.
    sentence_count: int = 0  # number of sentences in this paragraph
    sentences: List[Tuple[str, str]] = field(default_factory=list)  # List of (sentence_text, coords)
    
    def __reduce__(self):
        """Custom pickle support for dataclass."""
        return (
            self.__class__,
            (self.text, self.pdf_path, self.page_num, self.item_key, self.title, 
             self.section, self.sentence_count, self.sentences)
        )


@dataclass
class Answer:
    """Represents an extracted answer to a question."""
    text: str  # The answer text extracted from passage
    context: str  # Full paragraph context
    pdf_path: str
    page_num: int
    item_key: str
    title: str
    section: str = "body"
    start_char: int = 0  # Character position in context where answer starts
    end_char: int = 0  # Character position in context where answer ends
    score: float = 0.0  # QA model confidence score
    query: str = ""
    color: Tuple[float, float, float] = field(default_factory=lambda: (1, 1, 0))
    sentence_coords: List[str] = field(default_factory=list)  # TEI coordinates for highlighting
    retrieval_score: float = 0.0  # Semantic search distance/score
    
    def __reduce__(self):
        """Custom pickle support for dataclass."""
        return (
            self.__class__,
            (self.text, self.context, self.pdf_path, self.page_num, self.item_key, self.title,
             self.section, self.start_char, self.end_char, self.score, self.query, self.color, 
             self.sentence_coords, self.retrieval_score)
        )


class ZoteroRAG:
    
    # Reference section patterns to detect bibliography/references
    REFERENCE_PATTERNS = [
        r'^\s*references\s*$',
        r'^\s*bibliography\s*$',
        r'^\s*works\s+cited\s*$',
        r'^\s*literature\s+cited\s*$',
    ]
    
    # Section types to include in chunking (can be customized)
    CONTENT_SECTIONS = {
        'body': True,
        'abstract': True,
        'introduction': True,
        'conclusion': True,
        'results': True,
        'methods': True,
        'discussion': True,
    }

    # Serialize calls to GROBID to avoid exhausting its internal pool
    GROBID_LOCK = threading.Lock()
    
    # Patterns to identify non-content chunks
    NON_CONTENT_PATTERNS = [
        # Figure/Table captions and labels
        r'^(figure|fig\.?|table|tab\.?|tbl\.?)\s+\d+[:\.]',
        r'^(fig|figure|table|tbl)\s*\d+[:\.]?\s*[-–]',
        r'(figure|fig\.?|table|tab\.?|tbl\.?)\s+\d+[:\.]',
        
        # Short titles or headers (often all caps or title case)
        r'^[A-Z][A-Z\s]{5,80}$',
        r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s*$',
        
        # Page numbers and page references
        r'^\s*\d+\s*$',
        r'^page\s+\d+\s*$',
        r'^pp?\.?\s*\d+',
        
        # Very short text (likely headers/footers)
        r'^.{1,25}$',
        
        # Author/citation patterns (multiple names with commas/and)
        r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*(\s*,\s*[A-Z][a-z]+(\s+[A-Z][a-z]+)*)*(\s+and\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*)?$',
        
        # DOI/URL patterns (metadata)
        r'^\s*(doi|https?|www|url)[:\s]',
        
        # Copyright notices
        r'^©|^copyright|\(c\)',
        
        # Abstract/keywords markers
        r'^(abstract|keywords?|introduction|conclusion|acknowledgments?)\s*$',
        
        # Email/contact info
        r'[\w\.-]+@[\w\.-]+\.\w+',
    ]

    
    @staticmethod
    def _find_zotero_dir(custom_dir: str = None) -> str:
        """Find Zotero data directory"""
        if custom_dir and os.path.exists(custom_dir):
            return custom_dir
        possible_paths = [os.path.expanduser(p) for p in ["~/Zotero", "~/Documents/Zotero", "~/.zotero"]]
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'zotero.sqlite')):
                return path
        raise ValueError("Zotero directory not found. Please specify zotero_data_dir")
    
    @staticmethod
    def list_collections(zotero_data_dir: str = None) -> List[Dict]:
        """Load collections from the Zotero database."""
        zotero_dir = ZoteroRAG._find_zotero_dir(zotero_data_dir)
        db_path = os.path.join(zotero_dir, 'zotero.sqlite')
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()
        cursor.execute("SELECT collectionID, collectionName, parentCollectionID FROM collections ORDER BY collectionName")
        collections = [{'id': r[0], 'name': r[1], 'parent_id': r[2]} for r in cursor.fetchall()]
        conn.close()
        return collections
    
    def __init__(self, zotero_data_dir: str = None, model_name: str = "BAAI/bge-base-en-v1.5", 
                 collection_name: str = None, grobid_url: str = "http://localhost:8070", grobid_timeout: int = 180,
                 model_device: str = None, encode_batch_size: int = 8, tei_cache_dir: str = None,
                 output_base_dir: str = "output", qa_model: str = "deepset/roberta-base-squad2"):
        self.zotero_dir = self._find_zotero_dir(zotero_data_dir)
        self.storage_dir = os.path.join(self.zotero_dir, 'storage')
        self.db_path = os.path.join(self.zotero_dir, 'zotero.sqlite')
        self.collection_name = collection_name
        self.model_name = model_name
        self.qa_model_name = qa_model
        self.grobid_url = grobid_url
        self.grobid_timeout = grobid_timeout
        self.grobid_client = GrobidClient(grobid_server=self.grobid_url)
        self.device = model_device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.encode_batch_size = encode_batch_size
        self.qa_pipeline = None  # Lazy load on first use
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)

        # Persistent cache for TEI outputs keyed by PDF path+mtime
        base_cache = tei_cache_dir or os.path.join(output_base_dir, "tei_cache")
        coll_folder = self._sanitize_filename(self.collection_name)
        self.tei_cache_dir = os.path.join(base_cache, coll_folder)
        os.makedirs(self.tei_cache_dir, exist_ok=True)

        self.model = SentenceTransformer(model_name, device=self.device)
        
        self.paragraphs: List[Paragraph] = []
        self.index = None
        
        # Base directory for all indexes
        self.index_base_dir = output_base_dir
        
        # Collection-specific sub-directory
        self.collection_index_dir = os.path.join(
            self.index_base_dir, 
            self._sanitize_filename(self.collection_name)
        )
        
        self.query_colors = [
            (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.5, 0.0),
            (0.5, 1.0, 0.5), (1.0, 0.7, 0.8), (0.7, 0.5, 1.0),
        ]
        self.query_color_map = {}

    def _sanitize_filename(self, name: str) -> str:
        """Converts a string into a safe folder/file name."""
        if not name:
            return "_All_Library"
        s = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        s = re.sub(r'(?u)[^-\w.]', '', s)
        return s

    def _sanitize_model_name(self, model_name: str) -> str:
        """Convert model name to safe filename component."""
        # Extract last part of model path and sanitize
        model_short = model_name.split('/')[-1]
        return re.sub(r'[^a-zA-Z0-9_-]', '_', model_short)

    def set_index_paths(self, base_filename: str = None):
        """
        Set the file paths for the index and chunks inside the collection's folder.
        Includes model name in the filename.
        
        Args:
            base_filename: Optional base name. If None, auto-generates from collection and model.
        """
        if base_filename and os.path.dirname(base_filename):
            # Full path provided (from loading)
            self.index_path = f"{base_filename}.index"
            self.chunks_path = f"{base_filename}.pkl"
        else:
            # Generate filename with model info
            if base_filename is None:
                coll_name = self._sanitize_filename(self.collection_name)
                model_name = self._sanitize_model_name(self.model_name)
                base_filename = f"index_{model_name}"
            
            full_base_path = os.path.join(self.collection_index_dir, base_filename)
            self.index_path = f"{full_base_path}.index"
            self.chunks_path = f"{full_base_path}.pkl"
            
        logger.info(f"Index paths set to: {self.index_path} and {self.chunks_path}")

    def get_query_color(self, query: str) -> Tuple[float, float, float]:
        if query not in self.query_color_map:
            color_idx = len(self.query_color_map) % len(self.query_colors)
            self.query_color_map[query] = self.query_colors[color_idx]
        return self.query_color_map[query]
    
    def _load_qa_pipeline(self):
        """Lazily load the QA pipeline on first use."""
        if self.qa_pipeline is None:
            try:
                # Use transformers pipeline with the QA model
                self.qa_pipeline = pipeline(
                    'question-answering',
                    model=self.qa_model_name,
                    tokenizer=self.qa_model_name,
                    device=0 if self.device == "cuda" else -1
                )
            except Exception as e:
                logger.warning(f"Could not load QA model {self.qa_model_name}: {e}")
                self.qa_pipeline = False  # Mark as failed to avoid retrying
        return self.qa_pipeline if self.qa_pipeline else None
    
    def get_zotero_items(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()
        items = []
        if self.collection_name:
            cursor.execute("SELECT collectionID FROM collections WHERE collectionName = ?", (self.collection_name,))
            result = cursor.fetchone()
            if not result:
                conn.close(); raise ValueError(f"Collection '{self.collection_name}' not found.")
            collection_id = result[0]
            query = """
            SELECT a_items.key, ia.path, ia.parentItemID as sourceItemID FROM collectionItems ci
            JOIN itemAttachments ia ON ci.itemID = ia.parentItemID JOIN items a_items ON ia.itemID = a_items.itemID
            WHERE ci.collectionID = ? AND ia.contentType = 'application/pdf' AND ia.path IS NOT NULL
            """
            cursor.execute(query, (collection_id,))
        else:
            query = """
            SELECT i.key, ia.path, COALESCE(ia.parentItemID, i.itemID) as sourceItemID
            FROM items i JOIN itemAttachments ia ON i.itemID = ia.itemID
            WHERE ia.contentType = 'application/pdf' AND ia.path IS NOT NULL
            """
            cursor.execute(query)
        rows = cursor.fetchall()
        for key, path, src_id in rows:
            cursor.execute("SELECT v.value FROM itemData d JOIN itemDataValues v ON d.valueID = v.valueID JOIN fields f ON d.fieldID = f.fieldID WHERE d.itemID = ? AND f.fieldName = 'title'", (src_id,))
            title = (r[0] if (r := cursor.fetchone()) else "Unknown")
            pdf_path = os.path.join(self.storage_dir, key, path.replace('storage:', '')) if path and path.startswith('storage:') else path
            if pdf_path and os.path.exists(pdf_path):
                items.append({'key': key, 'path': pdf_path, 'title': title})
        conn.close()
        return items

    def _is_reference_section(self, text: str) -> bool:
        """Check if text appears to be start of reference/bibliography section."""
        text_lower = text.lower().strip()
        for pattern in self.REFERENCE_PATTERNS:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def _find_safe_batch_size(self, sample_texts: List[str], start_size: int = 2, max_size: int = 128) -> int:
        """Find largest safe batch size by trying increasingly larger sizes until OOM."""
        if not sample_texts:
            return start_size
        
        # Sample a small set to test with
        test_sample = sample_texts[:min(100, len(sample_texts))]
        
        current_size = start_size
        last_safe_size = start_size
        
        while current_size <= max_size:
            try:
                # Try encoding with current batch size
                with torch.no_grad():
                    _ = self.model.encode(
                        test_sample,
                        batch_size=current_size,
                        device=self.device
                    )
                last_safe_size = current_size
                current_size *= 2  # Double the batch size
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                # OOM hit; return last safe size
                # Check for various OOM indicators: "out of memory", "buffer size", "mps", "cuda"
                error_str = str(e).lower()
                if any(phrase in error_str for phrase in ["out of memory", "buffer size", "mps", "cuda", "memory"]):
                    return last_safe_size
                else:
                    # Different runtime error; continue testing
                    return last_safe_size
            except Exception as e:
                # Any other error; return last safe size
                return last_safe_size
        
        return last_safe_size

    def grobid_is_alive(self) -> bool:
        """Quick health check for the GROBID service."""
        try:
            resp = requests.get(f"{self.grobid_url}/api/isalive", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    
    def parse_pdf_with_grobid(self, pdf_path: str) -> Optional[ET.Element]:
        """Parse a single PDF using grobid-client-python and return TEI XML root."""
        try:
            if not self.grobid_is_alive():
                logger.error(f"GROBID not reachable at {self.grobid_url}")
                return None

            # Cache check: use pdf path + mtime to build stable key
            mtime = os.path.getmtime(pdf_path)
            cache_key = hashlib.md5(f"{pdf_path}:{mtime}".encode("utf-8")).hexdigest()
            cache_path = os.path.join(self.tei_cache_dir, f"{cache_key}.tei.xml")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        return ET.fromstring(f.read())
                except Exception:
                    pass  # fall through to reprocess if cache is unreadable

            with self.GROBID_LOCK:
                in_dir = tempfile.mkdtemp(prefix="grobid_in_")
                out_dir = tempfile.mkdtemp(prefix="grobid_out_")
                try:
                    base = os.path.basename(pdf_path)
                    temp_pdf = os.path.join(in_dir, base)
                    shutil.copy2(pdf_path, temp_pdf)

                    # Process with sentence segmentation and coordinates
                    self.grobid_client.process(
                        service="processFulltextDocument",
                        input_path=in_dir,
                        output=out_dir,
                        n=1,
                        tei_coordinates=True,
                        segment_sentences=True
                    )

                    expected = os.path.splitext(base)[0] + ".tei.xml"
                    tei_path = os.path.join(out_dir, expected)
                    if not os.path.exists(tei_path):
                        candidates = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".tei.xml")]
                        tei_path = candidates[0] if candidates else None

                    if tei_path and os.path.exists(tei_path):
                        with open(tei_path, "rb") as f:
                            content = f.read()
                        # Persist to cache for reuse
                        try:
                            with open(cache_path, "wb") as out_f:
                                out_f.write(content)
                        except Exception:
                            pass
                        return ET.fromstring(content)
                    else:
                        logger.warning(f"GROBID client did not produce TEI for {pdf_path}")
                        return None
                finally:
                    shutil.rmtree(in_dir, ignore_errors=True)
                    shutil.rmtree(out_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error parsing PDF with GROBID: {e}")
            return None
    
    def extract_paragraphs_from_tei(self, tei_root: ET.Element, pdf_path: str, item_title: str) -> List[Tuple[str, int, str, List[Tuple[str, str]]]]:
        """
        Extract paragraphs from TEI XML structure where each <p> is a paragraph.
        Returns list of (paragraph_text, page_number, section_type, sentences) tuples.
        sentences is a list of (sentence_text, coords) tuples.
        """
        paragraphs = []
        
        # Define TEI namespace
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        # Extract from abstract (each <p> is a paragraph)
        abstract = tei_root.find('.//tei:abstract', ns)
        if abstract is not None:
            # Iterate over <p> (paragraphs)
            for p_elem in abstract.findall('.//tei:p', ns):
                sentences_with_coords = []
                page_num = 0
                
                # Iterate over <s> (sentences) within this paragraph
                for s in p_elem.findall('.//tei:s', ns):
                    # Extract text from sentence
                    text_parts = []
                    for elem in s.iter():
                        if elem.text:
                            text_parts.append(elem.text)
                        if elem.tail:
                            text_parts.append(elem.tail)
                    
                    sentence_text = ''.join(text_parts).strip()
                    # Use coords from <s> element
                    coords = s.get('coords', '')
                    
                    if sentence_text:
                        sentences_with_coords.append((sentence_text, coords))
                    
                    # Try to extract page number from first sentence's coords
                    if page_num == 0 and coords:
                        try:
                            parts = coords.split(';')
                            if parts:
                                page_info = parts[0]
                                page_num = int(page_info.split(',')[0]) - 1
                        except:
                            pass
                
                # Join all sentences to form this paragraph
                if sentences_with_coords:
                    paragraph_text = ' '.join([sent for sent, _ in sentences_with_coords])
                    if len(paragraph_text.split()) >= 10:
                        paragraphs.append((paragraph_text, page_num, 'abstract', sentences_with_coords))
        
        # Extract from body (main content)
        body = tei_root.find('.//tei:body', ns)
        if body is not None:
            # Process all top-level div elements (sections)
            for section_div in body.findall('tei:div', ns):
                # Determine section type from head element
                head = section_div.find('tei:head', ns)
                section_type = 'body'
                if head is not None and head.text:
                    head_text = head.text.lower()
                    if 'abstract' in head_text:
                        section_type = 'abstract'
                    elif 'introduction' in head_text:
                        section_type = 'introduction'
                    elif 'method' in head_text or 'procedure' in head_text:
                        section_type = 'methods'
                    elif 'result' in head_text:
                        section_type = 'results'
                    elif 'discussion' in head_text:
                        section_type = 'discussion'
                    elif 'conclusion' in head_text:
                        section_type = 'conclusion'
                
                # Iterate over <p> (paragraphs) within this section
                for p_elem in section_div.findall('.//tei:p', ns):
                    sentences_with_coords = []
                    page_num = 0
                    
                    # Iterate over <s> (sentences) within this paragraph
                    for s in p_elem.findall('.//tei:s', ns):
                        # Extract text from sentence
                        text_parts = []
                        for elem in s.iter():
                            if elem.text:
                                text_parts.append(elem.text)
                            if elem.tail:
                                text_parts.append(elem.tail)
                        
                        sentence_text = ''.join(text_parts).strip()
                        # Use coords from <s> element
                        coords = s.get('coords', '')
                        
                        if sentence_text:
                            sentences_with_coords.append((sentence_text, coords))
                        
                        # Try to extract page number from first sentence's coords
                        if page_num == 0 and coords:
                            try:
                                parts = coords.split(';')
                                if parts:
                                    page_info = parts[0]
                                    page_num = int(page_info.split(',')[0]) - 1
                            except:
                                pass
                    
                    # Join all sentences from this <p> into one paragraph
                    if sentences_with_coords:
                        paragraph_text = ' '.join([sent for sent, _ in sentences_with_coords])
                        # Skip very short paragraphs
                        if len(paragraph_text.split()) >= 10:
                            paragraphs.append((paragraph_text, page_num, section_type, sentences_with_coords))
        
        return paragraphs
    
    def extract_text_chunks(self, pdf_path: str, item_title: str, chunk_size: int = None) -> List[Tuple[str, int, str]]:
        """
        Extract paragraphs from PDF using GROBID.
        Returns list of (paragraph_text, page_number, section_type) tuples.
        """
        tei_root = self.parse_pdf_with_grobid(pdf_path)
        if tei_root is None:
            logger.warning(f"GROBID parsing failed for {pdf_path}; no paragraphs extracted")
            return []

        paragraphs = self.extract_paragraphs_from_tei(tei_root, pdf_path, item_title)
        return paragraphs
    
    def _expand_to_sentences(self, paragraph: Paragraph, start_char: int, end_char: int) -> Tuple[str, int, int, List[str]]:
        """Expand answer span to include complete sentences and return their coordinates.
        
        Args:
            paragraph: The Paragraph object containing the answer
            start_char: Start position of answer in context
            end_char: End position of answer in context
            
        Returns:
            (expanded_text, new_start, new_end, sentence_coords) tuple
        """
        if not paragraph.sentences:
            return paragraph.text[start_char:end_char], start_char, end_char, []

        # Map character positions to sentences
        # We need to find which sentence(s) the answer span falls into
        
        start_sentence_idx = -1
        end_sentence_idx = -1
        
        current_pos = 0
        for i, (sent_text, _) in enumerate(paragraph.sentences):
            # Calculate range for this sentence including trailing space
            sent_len = len(sent_text)
            sent_start = current_pos
            sent_end = current_pos + sent_len
            
            # Check if answer start falls in this sentence
            if start_sentence_idx == -1 and start_char < sent_end + 1: # +1 for the space we add when joining
                start_sentence_idx = i
            
            # Check if answer end falls in this sentence
            if end_char <= sent_end + 1: # +1 for the space
                end_sentence_idx = i
                break
                
            current_pos += sent_len + 1 # +1 for space between sentences
            
        if start_sentence_idx == -1:
            start_sentence_idx = 0
        if end_sentence_idx == -1:
            end_sentence_idx = len(paragraph.sentences) - 1
            
        # Extract full text of all involved sentences
        involved_sentences = paragraph.sentences[start_sentence_idx : end_sentence_idx + 1]
        
        expanded_text = " ".join(s[0] for s in involved_sentences)
        sentence_coords = [s[1] for s in involved_sentences if s[1]]
        
        # Calculate new start/end relative to the whole paragraph text
        # (This is an approximation since we reconstruct paragraph text from sentences)
        new_start = 0
        for i in range(start_sentence_idx):
            new_start += len(paragraph.sentences[i][0]) + 1
            
        new_end = new_start + len(expanded_text)
        
        return expanded_text, new_start, new_end, sentence_coords

    def index_exists(self) -> bool:
        """Check if the index and chunks files exist for the current collection."""
        return os.path.exists(self.index_path) and os.path.exists(self.chunks_path)

    def load_index(self) -> int:
        """Load an existing FAISS index and chunks from disk."""
        if not self.index_path or not self.chunks_path:
            raise ValueError("Index paths are not set. Call set_index_paths() first.")
        if not self.index_exists():
            raise FileNotFoundError(f"Index file not found at {self.index_path}")
        
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.chunks_path, 'rb') as f:
                paragraphs_data = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            raise ValueError(f"Corrupted index files detected. Please rebuild the index. Error: {e}")
        
        # Convert back to Paragraph objects
        self.paragraphs = [
            Paragraph(
                text=item['text'],
                pdf_path=item['pdf_path'],
                page_num=item['page_num'],
                item_key=item['item_key'],
                title=item['title'],
                section=item.get('section', 'body'),
                sentence_count=item.get('sentence_count', 0),
                sentences=item.get('sentences', [])
            )
            for item in paragraphs_data
        ]
        return len(self.paragraphs)

    def build_index(self, force_rebuild: bool = False, progress_callback=None):
        """Build or load the FAISS index from Zotero PDFs.
        
        Args:
            progress_callback: Function that accepts (stage, current, total, message)
                              where stage is 'pdf' or 'encoding'
        """
        if not self.index_path or not self.chunks_path:
            raise ValueError("Index paths are not set. Call set_index_paths() first.")

        if not self.grobid_is_alive():
            raise ConnectionError(f"GROBID not reachable at {self.grobid_url}. Visit {self.grobid_url}/api/isalive or restart the container.")
        
        if not force_rebuild and self.index_exists():
            if progress_callback:
                progress_callback('pdf', 1, 1, "Loading existing index...")
            try:
                return self.load_index()
            except ValueError as e:
                if "Corrupted" in str(e):
                    # Index is corrupted, rebuild it
                    logger.warning(f"Detected corrupted index, rebuilding: {e}")
                    force_rebuild = True
                else:
                    raise

        items = self.get_zotero_items()
        if not items:
            raise ValueError("No PDF items found in the specified Zotero collection/library.")
        
        self.paragraphs = []
        all_texts = []
        
        # Stage 1: Process PDFs and extract paragraphs
        for idx, item in enumerate(items):
            if progress_callback:
                progress_callback('pdf', idx, len(items), f"Processing: {item['title'][:50]}...")
            
            paragraph_tuples = self.extract_text_chunks(item['path'], item['title'])
            for text, page_num, section, sentences in paragraph_tuples:
                # Filter by section type if needed
                if not self.CONTENT_SECTIONS.get(section, True):
                    continue
                sentence_count = len(sentences)
                paragraph = Paragraph(text, item['path'], page_num, item['key'], item['title'], section, sentence_count, sentences)
                self.paragraphs.append(paragraph)
                all_texts.append(text)
        
        if not all_texts:
            raise ValueError("No text could be extracted from the PDFs.")

        # Stage 2: Encode chunks with progress tracking
        if progress_callback:
            progress_callback('encoding', 0, len(all_texts), "Finding safe batch size...")
        
        # Auto-detect optimal batch size by testing
        detected_batch_size = self._find_safe_batch_size(all_texts, start_size=2, max_size=128)
        effective_batch_size = max(self.encode_batch_size, detected_batch_size) if self.encode_batch_size <= 8 else self.encode_batch_size
        
        if progress_callback:
            progress_callback('encoding', 0, len(all_texts), f"Encoding with batch size {effective_batch_size}...")
        
        # Manually batch and encode to show progress
        embeddings_list = []
        for i in range(0, len(all_texts), effective_batch_size):
            batch = all_texts[i:i + effective_batch_size]
            try:
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch,
                        show_progress_bar=False,
                        batch_size=effective_batch_size,
                        device=self.device
                    )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                # If we still hit OOM during actual encoding, reduce batch size further
                error_str = str(e).lower()
                if any(phrase in error_str for phrase in ["out of memory", "buffer size", "mps", "cuda", "memory"]):
                    # Fall back to even smaller batch size
                    fallback_size = max(1, effective_batch_size // 2)
                    if progress_callback:
                        progress_callback('encoding', i, len(all_texts), f"Reducing batch size to {fallback_size}...")
                    with torch.no_grad():
                        batch_embeddings = self.model.encode(
                            batch,
                            show_progress_bar=False,
                            batch_size=fallback_size,
                            device=self.device
                        )
                else:
                    raise
            
            embeddings_list.append(batch_embeddings)
            
            # Update progress after each batch
            processed = min(i + effective_batch_size, len(all_texts))
            if progress_callback:
                progress_callback('encoding', processed, len(all_texts), f"Encoded {processed}/{len(all_texts)} chunks...")
        
        embeddings = np.vstack(embeddings_list)
        
        # Update progress for index building
        if progress_callback:
            progress_callback('encoding', len(all_texts), len(all_texts), "Building index...")
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        faiss.write_index(self.index, self.index_path)
        # Save paragraphs as serializable format
        paragraphs_data = [
            {
                'text': para.text,
                'pdf_path': para.pdf_path,
                'page_num': para.page_num,
                'item_key': para.item_key,
                'title': para.title,
                'section': para.section,
                'sentence_count': para.sentence_count,
                'sentences': para.sentences
            }
            for para in self.paragraphs
        ]
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(paragraphs_data, f)
        
        return len(self.paragraphs)

    def answer_question(self, question: str, retrieval_threshold: float = 2.0, 
                        qa_score_threshold: float = 0.0, rerank_threshold: float = 0.25, 
                        progress_callback=None, rerank_callback=None) -> List[Answer]:
        """
        Answer a question using:
        1. FAISS Retrieval (Range Search)
        2. CrossEncoder Reranking (Threshold Filtering)
        3. QA Extraction (with Context Overlap/Sliding Window)
        """
        if not self.index: 
            raise ValueError("Index is not built.")
        
        # --- Stage 1: Retrieve candidate paragraphs (FAISS) ---
        query_embedding = self.model.encode([question], show_progress_bar=False)
        
        # Use range_search to get initial candidates
        lims, D, I = self.index.range_search(np.array(query_embedding).astype('float32'), retrieval_threshold)
        indices, distances = I[lims[0]:lims[1]], D[lims[0]:lims[1]]
        
        logger.debug(f"Question: {question}")
        logger.debug(f"Retrieved {len(indices)} paragraphs within L2 distance {retrieval_threshold}")
        
        candidate_data = []
        color = self.get_query_color(question)
        debug_candidates = []
        
        # Store tuple of (Paragraph, faiss_dist, ORIGINAL_INDEX)
        for idx, dist in zip(indices, distances):
            paragraph = self.paragraphs[idx]
            candidate_data.append((paragraph, float(dist), idx))
            debug_candidates.append({
                'paragraph': paragraph,
                'retrieval_score': float(dist),
                'kept': True
            })
        
        if not candidate_data:
            self.last_candidates = []
            return []
        
        # --- Stage 2: Rerank and Filter (CrossEncoder) ---
        
        # ** CALLBACK START: Pre-Rerank **
        if rerank_callback:
            rerank_callback(0, len(candidate_data), f"Cross-Encoding {len(candidate_data)} candidates...")
        
        # Prepare pairs for the reranker
        pairs = [[question, p[0].text] for p in candidate_data]
        
        # Predict scores
        raw_scores = self.reranker.predict(pairs)
        
        # Apply Sigmoid to convert logits to 0-1 probabilities
        probs = 1 / (1 + np.exp(-raw_scores))
        
        # Combine probabilities with candidate data
        scored_candidates = list(zip(probs, candidate_data))
        
        # FILTER: Keep only candidates above the rerank_threshold
        filtered_candidates = [
            (prob, item) for prob, item in scored_candidates 
            if prob >= rerank_threshold
        ]
        
        # SORT: By probability descending
        filtered_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Unpack back to simple list for QA loop: (paragraph, faiss_dist, original_idx)
        candidate_data = [item[1] for item in filtered_candidates]
        
        # ** CALLBACK END: Post-Rerank **
        if rerank_callback:
            rerank_callback(len(candidate_data), len(candidate_data), f"Reranking complete: {len(candidate_data)} paragraphs passed threshold.")

        logger.debug(f"Reranking: {len(scored_candidates)} -> {len(candidate_data)} paragraphs passed threshold {rerank_threshold}")

        # Update debug info
        self.last_candidates = [
            {
                'paragraph': d['paragraph'],
                'retrieval_score': d['retrieval_score'],
                'kept': d['paragraph'].text in [c[0].text for c in candidate_data],
            }
            for d in debug_candidates
        ]

        # --- Stage 3: Extract answers (QA Model + Sliding Window) ---
        qa_pipe = self._load_qa_pipeline()
        answers = []
        
        if qa_pipe:
            logger.debug(f"QA model available, extracting answers from {len(candidate_data)} paragraphs")
            try:
                for i, (paragraph, retrieval_score, original_idx) in enumerate(candidate_data):
                    if progress_callback:
                        progress_callback(i, len(candidate_data), f"QA Analysis: Paragraph {i+1}/{len(candidate_data)}")
                    
                    # --- CONTEXT OVERLAP LOGIC ---
                    prev_paragraph = None
                    combined_text = paragraph.text
                    shift_offset = 0
                    
                    if original_idx > 0:
                        potential_prev = self.paragraphs[original_idx - 1]
                        if potential_prev.pdf_path == paragraph.pdf_path:
                            prev_paragraph = potential_prev
                            combined_text = prev_paragraph.text + " " + paragraph.text
                            shift_offset = len(prev_paragraph.text) + 1
                    
                    QA_input = {'question': question, 'context': combined_text}
                    result = qa_pipe(**QA_input)
                    
                    if result:
                        if len(result['answer'].split()) < 3: 
                            continue 
                        
                        raw_answer = result['answer']
                        raw_start = result.get('start', 0)
                        raw_end = result.get('end', len(raw_answer))
                        
                        target_paragraph = paragraph
                        local_start = raw_start
                        local_end = raw_end
                        
                        if prev_paragraph and raw_end <= shift_offset:
                            target_paragraph = prev_paragraph
                        elif prev_paragraph and raw_start >= shift_offset:
                            target_paragraph = paragraph
                            local_start = raw_start - shift_offset
                            local_end = raw_end - shift_offset
                        elif prev_paragraph:
                            target_paragraph = paragraph
                            local_start = max(0, raw_start - shift_offset)
                            local_end = raw_end - shift_offset

                        expanded_text, new_start, new_end, sentence_coords = self._expand_to_sentences(
                            target_paragraph, local_start, local_end
                        )
                        
                        answer = Answer(
                            text=expanded_text,
                            context=target_paragraph.text,
                            pdf_path=target_paragraph.pdf_path,
                            page_num=target_paragraph.page_num,
                            item_key=target_paragraph.item_key,
                            title=target_paragraph.title,
                            section=target_paragraph.section,
                            start_char=new_start,
                            end_char=new_end,
                            score=float(result.get('score', 0.0)),
                            query=question,
                            color=color,
                            sentence_coords=sentence_coords,
                            retrieval_score=retrieval_score
                        )
                        answers.append(answer)
                
                if progress_callback:
                    progress_callback(len(candidate_data), len(candidate_data), "Finalizing results...")
                    
            except Exception as e:
                logger.error(f"QA extraction failed: {e}", exc_info=True)
                raise RuntimeError("QA extraction failed; aborting without fallback paragraphs")
        else:
            raise RuntimeError("QA pipeline not available; cannot answer question")
        
        # Sort by score descending first
        answers.sort(key=lambda x: x.score, reverse=True)
        
        # --- DEDUPLICATION ---
        # Keep unique answers based on (pdf_path, standardized_text)
        # This prevents the same answer appearing twice from overlapping chunks in the same PDF.
        unique_answers = []
        seen_answers = set()
        
        for ans in answers:
            if ans.score < qa_score_threshold:
                continue
                
            # Create a signature for the answer
            # We normalize text (lowercase + strip) to catch minor variations
            norm_text = " ".join(ans.text.lower().split())
            signature = (ans.pdf_path, norm_text)
            
            if signature not in seen_answers:
                seen_answers.add(signature)
                unique_answers.append(ans)
        
        return unique_answers

    def highlight_pdf(self, answers_for_pdf: List[Answer], output_path: str):
        """Highlight PDF using TEI sentence coordinates for precise highlighting."""
        if not answers_for_pdf: 
            return None
        
        try:
            import fitz
        except ImportError:
            logger.warning("PyMuPDF (fitz) not available for highlighting")
            return None
        
        try:
            # Use previously highlighted PDF if it exists (to preserve previous highlights),
            # otherwise use original PDF
            source_pdf = output_path if os.path.exists(output_path) else answers_for_pdf[0].pdf_path
            doc = fitz.open(source_pdf)
            answers_by_page = {}
            for answer in answers_for_pdf:
                answers_by_page.setdefault(answer.page_num, []).append(answer)
            
            for page_num, answers in answers_by_page.items():
                page = doc[page_num]
                for answer in answers:
                    highlighted_any = False

                    # Use TEI coordinates if available
                    if answer.sentence_coords:
                        for coords_str in answer.sentence_coords:
                            # Parse GROBID coords format: "page,x0,y0,width,height"
                            # Can have multiple coordinate groups separated by ';'
                            for coord_group in coords_str.split(';'):
                                try:
                                    parts = coord_group.split(',')
                                    if len(parts) >= 5:
                                        # GROBID uses 1-indexed pages
                                        coord_page = int(parts[0]) - 1
                                        if coord_page == page_num:
                                            x0 = float(parts[1])
                                            y0 = float(parts[2])
                                            width = float(parts[3])
                                            height = float(parts[4])
                                            # Convert to (x0, y0, x1, y1) format for PyMuPDF
                                            x1 = x0 + width
                                            y1 = y0 + height
                                            # Create rectangle for this text region
                                            rect = fitz.Rect(x0, y0, x1, y1)
                                            highlight = page.add_highlight_annot(rect)
                                            highlight.set_colors(stroke=answer.color)
                                            highlight.update()
                                            highlighted_any = True
                                except (ValueError, IndexError) as e:
                                    logger.debug(f"Could not parse coordinates '{coord_group}': {e}")
                                    continue
                        
                        if highlighted_any:
                            # Add annotation with question and answer
                            # Use first coordinate for annotation placement
                            try:
                                first_coords = answer.sentence_coords[0].split(';')[0]
                                parts = first_coords.split(',')
                                if len(parts) >= 5:
                                    x0, y0 = float(parts[1]), float(parts[2])
                                    answer_preview = answer.text[:100] + "..." if len(answer.text) > 100 else answer.text
                                    annot = page.add_text_annot(
                                        fitz.Point(x0, y0),
                                        f"Q: {answer.query[:80]}...\nA: {answer_preview}"
                                    )
                                    annot.update()
                            except:
                                logger.warning(f"Could not annotate {answer.pdf_path}.")
                                continue
                        else:
                            logger.warning(f"Could not highlight {answer.pdf_path} using coordinates on page {page_num}.")

                    if not highlighted_any:
                        logger.warning(f"Could not highlight {answer.pdf_path} using TEI coordinates on page {page_num}: {answer.text[:50]}... Skipping fallback search.")
            
            # Use incremental save if file already exists, otherwise normal save
            incremental = os.path.exists(output_path) and source_pdf == output_path
            try:
                doc.save(output_path, incremental=incremental)
            except RuntimeError as e:
                if "encryption" in str(e).lower():
                    doc.save(output_path, incremental=False)
                else:
                    raise
            doc.close()
            return output_path
        except Exception as e:
            logger.error(f"Error highlighting PDF: {e}", exc_info=True)
            return None