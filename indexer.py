"""FAISS index building and management."""

import os
import re
import logging
import pickle
from typing import List, Optional
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer

from models import Paragraph

logger = logging.getLogger(__name__)


class Indexer:
    """Manages FAISS index building and loading for semantic search."""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", 
                 device: str = None,
                 encode_batch_size: int = 8):
        """Initialize the indexer.
        
        Args:
            model_name: Name of the sentence transformer model.
            device: Device to use for encoding ('cpu', 'cuda', 'mps'). Auto-detect if None.
            encode_batch_size: Batch size for encoding.
        """
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.encode_batch_size = encode_batch_size
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index: Optional[faiss.Index] = None
        self.paragraphs: List[Paragraph] = []
        self.index_path: Optional[str] = None
        self.chunks_path: Optional[str] = None
    
    @staticmethod
    def _sanitize_model_name(model_name: str) -> str:
        """Convert model name to safe filename component."""
        model_short = model_name.split('/')[-1]
        return re.sub(r'[^a-zA-Z0-9_-]', '_', model_short)
    
    def set_index_paths(self, collection_name: str, output_base_dir: str = "output"):
        """Set the file paths for the index and chunks.
        
        Args:
            collection_name: Name of the collection (used for directory structure).
            output_base_dir: Base directory for storing indexes.
        """
        # Sanitize collection name for filesystem
        coll_folder = re.sub(r'(?u)[^-\w.]', '', 
                            collection_name.replace(" ", "_").replace("/", "_").replace("\\", "_")) \
                      if collection_name else "_All_Library"
        
        # Create collection-specific directory
        collection_index_dir = os.path.join(output_base_dir, coll_folder)
        os.makedirs(collection_index_dir, exist_ok=True)
        
        # Generate filename with model info
        model_name = self._sanitize_model_name(self.model_name)
        base_filename = f"index_{model_name}"
        
        full_base_path = os.path.join(collection_index_dir, base_filename)
        self.index_path = f"{full_base_path}.index"
        self.chunks_path = f"{full_base_path}.pkl"
        
        logger.info(f"Index paths set to: {self.index_path} and {self.chunks_path}")
    
    def index_exists(self) -> bool:
        """Check if the index and chunks files exist."""
        if not self.index_path or not self.chunks_path:
            return False
        return os.path.exists(self.index_path) and os.path.exists(self.chunks_path)
    
    def load_index(self) -> int:
        """Load an existing FAISS index and chunks from disk.
        
        Returns:
            Number of paragraphs loaded.
        """
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
    
    def _find_safe_batch_size(self, sample_texts: List[str], 
                              start_size: int = 2, 
                              max_size: int = 128,
                              target_memory_fraction: float = 0.75) -> int:
        """Find safe batch size targeting specific memory usage.
        
        Args:
            sample_texts: Sample of texts to test encoding with.
            start_size: Initial batch size to try.
            max_size: Maximum batch size to test.
            target_memory_fraction: Target fraction of memory to use (0.0-1.0).
            
        Returns:
            Safe batch size targeting the memory fraction.
        """
        if not sample_texts:
            return start_size
        
        # Sample a small set to test with
        test_sample = sample_texts[:min(100, len(sample_texts))]
        
        current_size = start_size
        last_safe_size = start_size
        
        while current_size <= max_size:
            try:
                with torch.no_grad():
                    _ = self.model.encode(
                        test_sample,
                        batch_size=current_size,
                        device=self.device,
                        show_progress_bar=False
                    )
                last_safe_size = current_size
                # Scale up more aggressively to find limit
                current_size = int(current_size * 1.5)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                error_str = str(e).lower()
                if any(phrase in error_str for phrase in 
                      ["out of memory", "buffer size", "mps", "cuda", "memory"]):
                    # Hit OOM, scale back to target fraction
                    return max(start_size, int(last_safe_size * target_memory_fraction))
                else:
                    return last_safe_size
            except Exception:
                return last_safe_size
        
        # Hit max size without OOM, use target fraction of max
        return max(start_size, int(last_safe_size * target_memory_fraction))
    
    def build_index(self, paragraphs: List[Paragraph], 
                   force_rebuild: bool = False, 
                   progress_callback=None) -> int:
        """Build FAISS index from paragraphs.
        
        Args:
            paragraphs: List of Paragraph objects to index.
            force_rebuild: If True, rebuild even if index exists.
            progress_callback: Function(stage, current, total, message) for progress updates.
            
        Returns:
            Number of paragraphs indexed.
        """
        if not self.index_path or not self.chunks_path:
            raise ValueError("Index paths are not set. Call set_index_paths() first.")
        
        # Check if index already exists
        if not force_rebuild and self.index_exists():
            if progress_callback:
                progress_callback('encoding', 1, 1, "Loading existing index...")
            try:
                return self.load_index()
            except ValueError as e:
                if "Corrupted" in str(e):
                    logger.warning(f"Detected corrupted index, rebuilding: {e}")
                    force_rebuild = True
                else:
                    raise
        
        if not paragraphs:
            raise ValueError("No paragraphs provided for indexing.")
        
        self.paragraphs = paragraphs
        all_texts = [p.text for p in paragraphs]
        
        # Stage 2: Encode chunks with progress tracking
        if self.encode_batch_size is None or self.encode_batch_size == 0:
            # Auto-detect safe batch size
            if progress_callback:
                progress_callback('encoding', 0, len(all_texts), "Auto-detecting safe batch size...")
            effective_batch_size = self._find_safe_batch_size(all_texts, start_size=2, max_size=128)
            logger.info(f"Auto-detected encoding batch size: {effective_batch_size}")
        else:
            effective_batch_size = self.encode_batch_size
        
        if progress_callback:
            progress_callback('encoding', 0, len(all_texts), 
                            f"Encoding with batch size {effective_batch_size}...")
        
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
                # If we still hit OOM, reduce batch size further
                error_str = str(e).lower()
                if any(phrase in error_str for phrase in 
                      ["out of memory", "buffer size", "mps", "cuda", "memory"]):
                    fallback_size = max(1, effective_batch_size // 2)
                    if progress_callback:
                        progress_callback('encoding', i, len(all_texts), 
                                        f"Reducing batch size to {fallback_size}...")
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
                progress_callback('encoding', processed, len(all_texts), 
                                f"Encoded {processed}/{len(all_texts)} chunks...")
        
        embeddings = np.vstack(embeddings_list)
        
        # Build index
        if progress_callback:
            progress_callback('encoding', len(all_texts), len(all_texts), "Building index...")
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Save to disk
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
        
        logger.info(f"Index built with {len(self.paragraphs)} paragraphs")
        return len(self.paragraphs)
    
    def search(self, query: str, threshold: float = 2.0) -> List[tuple]:
        """Search the index for relevant paragraphs.
        
        Args:
            query: Query string.
            threshold: L2 distance threshold for range search.
            
        Returns:
            List of (Paragraph, distance, original_index) tuples.
        """
        if not self.index:
            raise ValueError("Index is not built or loaded.")
        
        query_embedding = self.model.encode([query], show_progress_bar=False)
        
        # Use range_search to get candidates within threshold
        lims, D, I = self.index.range_search(
            np.array(query_embedding).astype('float32'), 
            threshold
        )
        indices, distances = I[lims[0]:lims[1]], D[lims[0]:lims[1]]
        
        results = []
        for idx, dist in zip(indices, distances):
            paragraph = self.paragraphs[idx]
            results.append((paragraph, float(dist), int(idx)))
        
        return results
