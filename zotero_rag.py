import os
import sqlite3
import re
import threading
import hashlib
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
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

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

@dataclass
class Chunk:
    text: str
    pdf_path: str
    page_num: int
    item_key: str
    title: str
    section: str = "body"  # section type: body, abstract, intro, etc.
    bbox: Tuple[float, float, float, float] = None
    query: str = ""
    color: Tuple[float, float, float] = field(default_factory=lambda: (1, 1, 0))
    
    def __reduce__(self):
        """Custom pickle support for dataclass."""
        return (
            self.__class__,
            (self.text, self.pdf_path, self.page_num, self.item_key, self.title, 
             self.section, self.bbox, self.query, self.color)
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
    
    def __init__(self, zotero_data_dir: str = None, model_name: str = "BAAI/bge-small-en-v1.5", 
                 collection_name: str = None, grobid_url: str = "http://localhost:8070", grobid_timeout: int = 180,
                 model_device: str = None, encode_batch_size: int = 8, tei_cache_dir: str = None,
                 output_base_dir: str = "output"):
        self.zotero_dir = self._find_zotero_dir(zotero_data_dir)
        self.storage_dir = os.path.join(self.zotero_dir, 'storage')
        self.db_path = os.path.join(self.zotero_dir, 'zotero.sqlite')
        self.collection_name = collection_name
        self.model_name = model_name
        self.grobid_url = grobid_url
        self.grobid_timeout = grobid_timeout
        self.grobid_client = GrobidClient(grobid_server=self.grobid_url)
        self.device = model_device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.encode_batch_size = encode_batch_size

        # Persistent cache for TEI outputs keyed by PDF path+mtime
        base_cache = tei_cache_dir or os.path.join(output_base_dir, "tei_cache")
        coll_folder = self._sanitize_filename(self.collection_name)
        self.tei_cache_dir = os.path.join(base_cache, coll_folder)
        os.makedirs(self.tei_cache_dir, exist_ok=True)

        self.model = SentenceTransformer(model_name, device=self.device)
        
        self.chunks: List[Chunk] = []
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
            
        print(f"Index paths set to: {self.index_path} and {self.chunks_path}")

    def get_query_color(self, query: str) -> Tuple[float, float, float]:
        if query not in self.query_color_map:
            color_idx = len(self.query_color_map) % len(self.query_colors)
            self.query_color_map[query] = self.query_colors[color_idx]
        return self.query_color_map[query]
    
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
                print(f"GROBID not reachable at {self.grobid_url}")
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
                        print(f"GROBID client did not produce TEI for {pdf_path}")
                        return None
                finally:
                    shutil.rmtree(in_dir, ignore_errors=True)
                    shutil.rmtree(out_dir, ignore_errors=True)
        except Exception as e:
            print(f"Error parsing PDF with GROBID: {e}")
            return None
    
    def extract_sentences_from_tei(self, tei_root: ET.Element, pdf_path: str, item_title: str) -> List[Tuple[str, int, str]]:
        """
        Extract sentences from TEI XML structure.
        Returns list of (sentence_text, page_number, section_type) tuples.
        """
        sentences = []
        
        # Define TEI namespace
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        # Extract from body (main content)
        body = tei_root.find('.//tei:body', ns)
        if body is not None:
            # Process all div elements (sections)
            for div in body.findall('.//tei:div', ns):
                # Determine section type from head element
                head = div.find('tei:head', ns)
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
                
                # Extract sentences from paragraphs
                for p in div.findall('tei:p', ns):
                    for s in p.findall('tei:s', ns):
                        # Extract text from sentence
                        text_parts = []
                        for elem in s.iter():
                            if elem.text:
                                text_parts.append(elem.text)
                            if elem.tail:
                                text_parts.append(elem.tail)
                        
                        sentence_text = ''.join(text_parts).strip()
                        
                        # Skip very short sentences and non-content
                        if len(sentence_text.split()) >= 3:
                            # Try to extract page number from coords if available
                            page_num = 0
                            coords = s.get('coords')
                            if coords:
                                try:
                                    # coords format: "page_x,page_y,page_width,page_height"
                                    parts = coords.split(';')
                                    if parts:
                                        page_info = parts[0]
                                        page_num = int(page_info.split(',')[0]) - 1  # Convert to 0-indexed
                                except:
                                    pass
                            
                            sentences.append((sentence_text, page_num, section_type))
        
        return sentences
    
    def extract_text_chunks(self, pdf_path: str, item_title: str, chunk_size: int = None) -> List[Tuple[str, int, Tuple, str]]:
        """
        Extract text chunks (sentences) from PDF using GROBID.
        Returns list of (sentence_text, page_number, bbox, section_type) tuples.
        """
        chunks = []
        
        
        tei_root = self.parse_pdf_with_grobid(pdf_path)
        if tei_root is None:
            print(f"GROBID parsing failed for {pdf_path}; no chunks extracted")
            return []

        sentences = self.extract_sentences_from_tei(tei_root, pdf_path, item_title)
        for sentence_text, page_num, section_type in sentences:
            chunks.append((sentence_text, page_num, None, section_type))
        return chunks
        
    def _extract_keywords(self, query: str) -> Set[str]:
        """Extract meaningful keywords from query (simple approach)."""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'can', 'about', 'what',
                     'which', 'who', 'when', 'where', 'why', 'how'}
        
        # Extract words, convert to lowercase, remove punctuation
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = {w for w in words if w not in stop_words and len(w) > 2}
        return keywords

    def _chunk_contains_keywords(self, chunk_text: str, keywords: Set[str]) -> bool:
        """Check if chunk contains at least one keyword from the query."""
        if not keywords:
            return True  # If no keywords extracted, don't filter
        
        chunk_lower = chunk_text.lower()
        return any(keyword in chunk_lower for keyword in keywords)

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
                chunks_data = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            raise ValueError(f"Corrupted index files detected. Please rebuild the index. Error: {e}")
        
        # Convert back to Chunk objects
        self.chunks = [
            Chunk(
                text=item['text'],
                pdf_path=item['pdf_path'],
                page_num=item['page_num'],
                item_key=item['item_key'],
                title=item['title'],
                section=item.get('section', 'body'),
                bbox=item.get('bbox'),
                query=item.get('query', ''),
                color=tuple(item.get('color', (1, 1, 0)))
            )
            for item in chunks_data
        ]
        return len(self.chunks)

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
                    print(f"Detected corrupted index, rebuilding: {e}")
                    force_rebuild = True
                else:
                    raise

        items = self.get_zotero_items()
        if not items:
            raise ValueError("No PDF items found in the specified Zotero collection/library.")
        
        self.chunks = []
        all_texts = []
        
        # Stage 1: Process PDFs
        for idx, item in enumerate(items):
            if progress_callback:
                progress_callback('pdf', idx, len(items), f"Processing: {item['title'][:50]}...")
            
            text_chunks = self.extract_text_chunks(item['path'], item['title'])
            for text, page_num, bbox, section in text_chunks:
                # Filter by section type if needed
                if not self.CONTENT_SECTIONS.get(section, True):
                    continue
                chunk = Chunk(text, item['path'], page_num, item['key'], item['title'], section, bbox)
                self.chunks.append(chunk)
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
        # Save chunks as serializable format
        chunks_data = [
            {
                'text': chunk.text,
                'pdf_path': chunk.pdf_path,
                'page_num': chunk.page_num,
                'item_key': chunk.item_key,
                'title': chunk.title,
                'section': chunk.section,
                'bbox': chunk.bbox,
                'query': chunk.query,
                'color': chunk.color
            }
            for chunk in self.chunks
        ]
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(chunks_data, f)
        
        return len(self.chunks)

    def search(self, query: str, threshold: float = 1.2) -> List[Tuple[Chunk, float]]:
        if not self.index: raise ValueError("Index is not built.")
        
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        query_embedding = self.model.encode([query])
        lims, D, I = self.index.range_search(np.array(query_embedding).astype('float32'), threshold)
        indices, distances = I[lims[0]:lims[1]], D[lims[0]:lims[1]]
        results = []
        color = self.get_query_color(query)
        
        for idx, dist in zip(indices, distances):
            chunk = self.chunks[idx]
            # Filter by keyword presence
            if self._chunk_contains_keywords(chunk.text, keywords):
                results.append((Chunk(chunk.text, chunk.pdf_path, chunk.page_num, chunk.item_key, 
                                     chunk.title, chunk.section, chunk.bbox, query, color), float(dist)))
        
        results.sort(key=lambda x: x[1])
        return results

    def highlight_pdf(self, chunks_for_pdf: List[Chunk], output_path: str):
        """Highlight PDF by finding and marking sentences with query match. Preserves previous highlights."""
        if not chunks_for_pdf: 
            return None
        
        try:
            import fitz
        except ImportError:
            print("PyMuPDF (fitz) not available for highlighting")
            return None
        
        try:
            # Use previously highlighted PDF if it exists (to preserve previous highlights),
            # otherwise use original PDF
            source_pdf = output_path if os.path.exists(output_path) else chunks_for_pdf[0].pdf_path
            doc = fitz.open(source_pdf)
            chunks_by_page = {}
            for chunk in chunks_for_pdf:
                chunks_by_page.setdefault(chunk.page_num, []).append(chunk)
            
            for page_num, chunks in chunks_by_page.items():
                page = doc[page_num]
                for chunk in chunks:
                    # Search for the sentence in the PDF
                    # Try full sentence first, then progressively shorter versions
                    search_text = chunk.text
                    areas = []
                    
                    # Try increasingly shorter versions of the text
                    for attempt in range(3):
                        if attempt == 0:
                            search_text = chunk.text[:100]
                        elif attempt == 1:
                            search_text = " ".join(chunk.text.split()[:10])
                        else:
                            search_text = " ".join(chunk.text.split()[:5])
                        
                        try:
                            areas = page.search_for(search_text)
                            if areas:
                                break
                        except:
                            pass
                    
                    # Highlight found areas
                    for area in areas:
                        highlight = page.add_highlight_annot(area)
                        highlight.set_colors(stroke=chunk.color)
                        highlight.update()
                    
                    # Add annotation with query info
                    if areas:
                        annot = page.add_text_annot(areas[0].tl, f"Q: {chunk.query[:50]}...")
                        annot.update()
            
            doc.save(output_path)
            doc.close()
            return output_path
        except Exception as e:
            print(f"Error highlighting PDF: {e}")
            return None