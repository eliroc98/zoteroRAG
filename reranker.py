"""Cross-encoder reranking model."""

import logging
from typing import List, Tuple
import numpy as np
from sentence_transformers import CrossEncoder

from models import Paragraph

logger = logging.getLogger(__name__)


class Reranker:
    """Handles reranking of retrieved candidates using a cross-encoder model."""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', 
                 device: str = None,
                 batch_size: int = None):
        """Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model.
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detect if None.
            batch_size: Batch size for reranking. If None, auto-detect safe size.
        """
        import torch
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.batch_size = batch_size  # None means auto-detect
        self.model = CrossEncoder(model_name, device=self.device)
        logger.info(f"Reranker initialized with model {model_name} on {self.device}")
    
    def adaptive_rerank_threshold(self, rerank_scores: np.ndarray, base_threshold: float = 0.25) -> float:
        """Adjust rerank threshold based on score distribution.
        
        Args:
            rerank_scores: Array of rerank probability scores (0-1).
            base_threshold: The baseline threshold to adjust from.
            
        Returns:
            Adjusted threshold value.
        """
        if len(rerank_scores) < 3:
            return base_threshold
        
        # Calculate statistics
        mean_score = np.mean(rerank_scores)
        std_score = np.std(rerank_scores)
        max_score = np.max(rerank_scores)
        
        # If there's a clear winner (high max, low mean), be more selective
        if max_score > 0.7 and mean_score < 0.4:
            adjusted = max(base_threshold, mean_score + 0.5 * std_score)
            logger.info(f"Adaptive threshold: Clear winner detected (max={max_score:.3f}, mean={mean_score:.3f}) -> {adjusted:.3f}")
            return adjusted
        
        # If scores are uniformly low, be more lenient
        if max_score < 0.4:
            adjusted = min(base_threshold, base_threshold * 0.7)
            logger.info(f"Adaptive threshold: Uniformly low scores (max={max_score:.3f}) -> {adjusted:.3f}")
            return adjusted
        
        # If scores are uniformly high, be more selective
        if mean_score > 0.6:
            adjusted = max(base_threshold, mean_score - 0.5 * std_score)
            logger.info(f"Adaptive threshold: Uniformly high scores (mean={mean_score:.3f}) -> {adjusted:.3f}")
            return adjusted
        
        logger.debug(f"Adaptive threshold: Using base threshold {base_threshold:.3f} (mean={mean_score:.3f}, max={max_score:.3f})")
        return base_threshold
    
    def _find_safe_batch_size(self, pairs: List[List[str]], 
                               start_size: int = 2, 
                               max_size: int = 128,
                               target_memory_fraction: float = 0.75) -> int:
        """Find safe batch size targeting specific memory usage.
        
        Args:
            pairs: Sample of text pairs to test with.
            start_size: Initial batch size to try.
            max_size: Maximum batch size to test.
            target_memory_fraction: Target fraction of memory to use (0.0-1.0).
            
        Returns:
            Safe batch size.
        """
        import torch
        test_sample = pairs[:min(100, len(pairs))]
        
        current_size = start_size
        last_safe_size = start_size
        
        while current_size <= max_size:
            try:
                _ = self.model.predict(test_sample[:current_size], show_progress_bar=False)
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
    
    def rerank(self, query: str, 
               candidates: List[Tuple[Paragraph, float, int]], 
               threshold: float = 0.25,
               progress_callback=None) -> List[Tuple[Paragraph, float, int, float]]:
        """Rerank candidates using cross-encoder scores.
        
        Args:
            query: The query string.
            candidates: List of (Paragraph, retrieval_score, original_index) tuples.
            threshold: Minimum probability threshold to keep candidates.
            progress_callback: Function(current, total, message) for progress updates.
            
        Returns:
            List of (Paragraph, retrieval_score, original_index, rerank_score) tuples,
            filtered and sorted by rerank_score descending.
        """
        if not candidates:
            return []
        
        # Prepare pairs for the reranker
        pairs = [[query, p[0].text] for p in candidates]
        
        # Determine batch size
        if self.batch_size is None:
            # Auto-detect safe batch size
            if progress_callback:
                progress_callback(0, len(candidates), "Auto-detecting safe batch size...")
            effective_batch_size = self._find_safe_batch_size(pairs, start_size=2, max_size=128)
            logger.info(f"Auto-detected reranker batch size: {effective_batch_size}")
        else:
            effective_batch_size = self.batch_size
        
        if progress_callback:
            progress_callback(0, len(candidates), 
                            f"Cross-Encoding {len(candidates)} candidates (batch size: {effective_batch_size})...")
        
        # Predict scores in batches with progress tracking
        all_scores = []
        num_batches = (len(pairs) + effective_batch_size - 1) // effective_batch_size
        
        for batch_idx, i in enumerate(range(0, len(pairs), effective_batch_size)):
            batch_pairs = pairs[i:i + effective_batch_size]
            
            # Update progress before processing batch
            if progress_callback:
                progress_callback(i, len(pairs), 
                                f"Processing batch {batch_idx + 1}/{num_batches}...")
            
            batch_scores = self.model.predict(batch_pairs, show_progress_bar=False)
            all_scores.extend(batch_scores)
            
            # Update progress after processing batch
            if progress_callback:
                processed = min(i + effective_batch_size, len(pairs))
                progress_callback(processed, len(pairs), 
                                f"Completed {processed}/{len(pairs)} candidates...")
        
        raw_scores = np.array(all_scores)
        
        # Apply Sigmoid to convert logits to 0-1 probabilities
        probs = 1 / (1 + np.exp(-raw_scores))
        
        # Adapt threshold based on score distribution
        adjusted_threshold = self.adaptive_rerank_threshold(probs, threshold)
        
        # Combine probabilities with candidate data
        # Each item: (paragraph, retrieval_score, original_idx, rerank_score)
        scored_candidates = [
            (p[0], p[1], p[2], float(prob))
            for p, prob in zip(candidates, probs)
        ]
        
        # Filter by adjusted threshold
        filtered_candidates = [
            item for item in scored_candidates 
            if item[3] >= adjusted_threshold
        ]
        
        # Sort by rerank score descending
        filtered_candidates.sort(key=lambda x: x[3], reverse=True)
        
        if progress_callback:
            progress_callback(len(candidates), len(candidates), 
                            f"Reranking complete: {len(filtered_candidates)} paragraphs passed threshold.")
        
        # Log threshold statistics
        if adjusted_threshold != threshold:
            logger.info(f"Reranking: {len(candidates)} -> {len(filtered_candidates)} paragraphs "
                       f"(base threshold: {threshold:.3f}, adjusted: {adjusted_threshold:.3f})")
        else:
            logger.debug(f"Reranking: {len(candidates)} -> {len(filtered_candidates)} "
                        f"paragraphs passed threshold {threshold:.3f}")
        
        return filtered_candidates
