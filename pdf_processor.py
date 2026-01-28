"""PDF processing using GROBID service."""

import os
import hashlib
import logging
import threading
import tempfile
import shutil
import re
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
import requests
from grobid_client.grobid_client import GrobidClient

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF parsing and text extraction using GROBID."""
    
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
    
    def __init__(self, grobid_url: str = "http://localhost:8070", 
                 grobid_timeout: int = 180, 
                 tei_cache_dir: str = None):
        """Initialize PDF processor.
        
        Args:
            grobid_url: URL of the GROBID service.
            grobid_timeout: Timeout in seconds for GROBID requests.
            tei_cache_dir: Directory to cache TEI XML outputs.
        """
        self.grobid_url = grobid_url
        self.grobid_timeout = grobid_timeout
        self.grobid_client = None  # Lazy initialization only when needed
        self.tei_cache_dir = tei_cache_dir or "tei_cache"
        os.makedirs(self.tei_cache_dir, exist_ok=True)
    
    def is_alive(self) -> bool:
        """Quick health check for the GROBID service."""
        try:
            resp = requests.get(f"{self.grobid_url}/api/isalive", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    
    def parse_pdf(self, pdf_path: str) -> Optional[ET.Element]:
        """Parse a single PDF using GROBID and return TEI XML root.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            XML Element root of the TEI document, or None if parsing failed.
        """
        try:
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
            
            # Only check GROBID availability if we need to parse (cache miss)
            if not self.is_alive():
                logger.error(f"GROBID not reachable at {self.grobid_url}")
                return None
            
            # Lazy initialization of GROBID client only when needed
            if self.grobid_client is None:
                self.grobid_client = GrobidClient(grobid_server=self.grobid_url)

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
                        candidates = [
                            os.path.join(out_dir, f) 
                            for f in os.listdir(out_dir) 
                            if f.endswith(".tei.xml")
                        ]
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
    
    def extract_paragraphs_from_tei(self, tei_root: ET.Element, 
                                     pdf_path: str, 
                                     item_title: str) -> List[Tuple[str, int, str, List[Tuple[str, str]]]]:
        """Extract paragraphs from TEI XML structure.
        
        Args:
            tei_root: Root element of TEI XML.
            pdf_path: Path to the original PDF (for metadata).
            item_title: Title of the document (for metadata).
            
        Returns:
            List of (paragraph_text, page_number, section_type, sentences) tuples.
            sentences is a list of (sentence_text, coords) tuples.
        """
        paragraphs = []
        
        # Define TEI namespace
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        # Extract from abstract (each <p> is a paragraph)
        abstract = tei_root.find('.//tei:abstract', ns)
        if abstract is not None:
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
                        text_parts = []
                        for elem in s.iter():
                            if elem.text:
                                text_parts.append(elem.text)
                            if elem.tail:
                                text_parts.append(elem.tail)
                        
                        sentence_text = ''.join(text_parts).strip()
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
                        if len(paragraph_text.split()) >= 10:
                            paragraphs.append((paragraph_text, page_num, section_type, sentences_with_coords))
        
        return paragraphs
    
    def extract_text_chunks(self, pdf_path: str, 
                           item_title: str) -> List[Tuple[str, int, str, List[Tuple[str, str]]]]:
        """Extract paragraphs from PDF using GROBID.
        
        Args:
            pdf_path: Path to the PDF file.
            item_title: Title of the document.
            
        Returns:
            List of (paragraph_text, page_number, section_type, sentences) tuples.
        """
        tei_root = self.parse_pdf(pdf_path)
        if tei_root is None:
            logger.warning(f"GROBID parsing failed for {pdf_path}; no paragraphs extracted")
            return []

        paragraphs = self.extract_paragraphs_from_tei(tei_root, pdf_path, item_title)
        return paragraphs
