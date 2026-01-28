"""Question answering engine using extractive QA models."""

import logging
from typing import List, Tuple, Optional
from transformers import pipeline

from models import Paragraph, Answer

logger = logging.getLogger(__name__)


class QAEngine:
    """Handles extractive question answering over text passages."""
    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2", 
                 device: str = None):
        """Initialize the QA engine.
        
        Args:
            model_name: Name of the HuggingFace QA model.
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detect if None.
        """
        import torch
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.pipeline = None
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the QA pipeline."""
        try:
            device_id = 0 if self.device == "cuda" else -1
            self.pipeline = pipeline(
                'question-answering',
                model=self.model_name,
                tokenizer=self.model_name,
                device=device_id
            )
            logger.info(f"QA pipeline loaded: {self.model_name} on device {self.device}")
        except Exception as e:
            logger.warning(f"Could not load QA model {self.model_name}: {e}")
            self.pipeline = None
    
    def _expand_to_sentences(self, paragraph: Paragraph, 
                            start_char: int, 
                            end_char: int) -> Tuple[str, int, int, List[str]]:
        """Expand answer span to include complete sentences and return their coordinates.
        
        Args:
            paragraph: The Paragraph object containing the answer.
            start_char: Start position of answer in context.
            end_char: End position of answer in context.
            
        Returns:
            (expanded_text, new_start, new_end, sentence_coords) tuple.
        """
        if not paragraph.sentences:
            return paragraph.text[start_char:end_char], start_char, end_char, []

        # Map character positions to sentences
        start_sentence_idx = -1
        end_sentence_idx = -1
        
        current_pos = 0
        for i, (sent_text, _) in enumerate(paragraph.sentences):
            sent_len = len(sent_text)
            sent_start = current_pos
            sent_end = current_pos + sent_len
            
            # Check if answer start falls in this sentence
            if start_sentence_idx == -1 and start_char < sent_end + 1:
                start_sentence_idx = i
            
            # Check if answer end falls in this sentence
            if end_char <= sent_end + 1:
                end_sentence_idx = i
                break
                
            current_pos += sent_len + 1  # +1 for space between sentences
            
        if start_sentence_idx == -1:
            start_sentence_idx = 0
        if end_sentence_idx == -1:
            end_sentence_idx = len(paragraph.sentences) - 1
            
        # Extract full text of all involved sentences
        involved_sentences = paragraph.sentences[start_sentence_idx : end_sentence_idx + 1]
        
        expanded_text = " ".join(s[0] for s in involved_sentences)
        sentence_coords = [s[1] for s in involved_sentences if s[1]]
        
        # Calculate new start/end relative to the whole paragraph text
        new_start = 0
        for i in range(start_sentence_idx):
            new_start += len(paragraph.sentences[i][0]) + 1
            
        new_end = new_start + len(expanded_text)
        
        return expanded_text, new_start, new_end, sentence_coords
    
    def extract_answers(self, question: str, 
                       candidates: List[Tuple[Paragraph, float, int, float]],
                       all_paragraphs: List[Paragraph],
                       qa_score_threshold: float = 0.0,
                       color: Tuple[float, float, float] = (1, 1, 0),
                       progress_callback=None) -> List[Answer]:
        """Extract answers from candidate paragraphs using QA model.
        
        Args:
            question: The question to answer.
            candidates: List of (Paragraph, retrieval_score, original_index, rerank_score) tuples.
            all_paragraphs: Full list of all paragraphs (for context overlap).
            qa_score_threshold: Minimum QA confidence score threshold.
            color: RGB color tuple for highlighting.
            progress_callback: Function(current, total, message) for progress updates.
            
        Returns:
            List of Answer objects, deduplicated and sorted by score.
        """
        if not self.pipeline:
            raise RuntimeError("QA pipeline not available; cannot answer question")
        
        if not candidates:
            return []
        
        answers = []
        
        for i, (paragraph, retrieval_score, original_idx, rerank_score) in enumerate(candidates):
            if progress_callback:
                progress_callback(i, len(candidates), 
                                f"QA Analysis: Paragraph {i+1}/{len(candidates)}")
            
            # --- CONTEXT OVERLAP LOGIC ---
            # Check if there's a previous paragraph from the same document
            prev_paragraph = None
            combined_text = paragraph.text
            shift_offset = 0
            
            if original_idx > 0:
                potential_prev = all_paragraphs[original_idx - 1]
                if potential_prev.pdf_path == paragraph.pdf_path:
                    prev_paragraph = potential_prev
                    combined_text = prev_paragraph.text + " " + paragraph.text
                    shift_offset = len(prev_paragraph.text) + 1
            
            # Run QA model
            qa_input = {'question': question, 'context': combined_text}
            result = self.pipeline(**qa_input)
            
            if result:
                # Filter very short answers
                if len(result['answer'].split()) < 3:
                    continue
                
                raw_answer = result['answer']
                raw_start = result.get('start', 0)
                raw_end = result.get('end', len(raw_answer))
                
                # Determine which paragraph the answer belongs to
                target_paragraph = paragraph
                local_start = raw_start
                local_end = raw_end
                
                if prev_paragraph and raw_end <= shift_offset:
                    # Answer is entirely in previous paragraph
                    target_paragraph = prev_paragraph
                elif prev_paragraph and raw_start >= shift_offset:
                    # Answer is entirely in current paragraph
                    target_paragraph = paragraph
                    local_start = raw_start - shift_offset
                    local_end = raw_end - shift_offset
                elif prev_paragraph:
                    # Answer spans both paragraphs; use current paragraph
                    target_paragraph = paragraph
                    local_start = max(0, raw_start - shift_offset)
                    local_end = raw_end - shift_offset

                # Expand to sentence boundaries
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
            progress_callback(len(candidates), len(candidates), "Finalizing results...")
        
        # Sort by QA score descending
        answers.sort(key=lambda x: x.score, reverse=True)
        
        # --- DEDUPLICATION ---
        # Keep unique answers based on (pdf_path, normalized_text)
        unique_answers = []
        seen_answers = set()
        
        for ans in answers:
            if ans.score < qa_score_threshold:
                continue
                
            # Normalize text to catch minor variations
            norm_text = " ".join(ans.text.lower().split())
            signature = (ans.pdf_path, norm_text)
            
            if signature not in seen_answers:
                seen_answers.add(signature)
                unique_answers.append(ans)
        
        logger.info(f"Extracted {len(unique_answers)} unique answers from {len(candidates)} candidates")
        return unique_answers
