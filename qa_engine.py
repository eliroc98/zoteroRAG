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
    
    def classify_question_type(self, question: str) -> str:
        """Classify the type of question being asked.
        
        Args:
            question: The question text.
            
        Returns:
            Question type: 'factoid', 'explanation', 'comparison', 'definition', or 'general'.
        """
        question_lower = question.lower().strip()
        
        # Definition questions
        if 'define' in question_lower or 'what is' in question_lower or 'what are' in question_lower:
            return 'definition'
        
        # Factoid questions
        if any(question_lower.startswith(w) for w in ['who ', 'what ', 'when ', 'where ']):
            return 'factoid'
        
        # Comparison/list questions
        if any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs', 'list ']):
            return 'comparison'
        
        # Explanation questions
        if any(question_lower.startswith(w) for w in ['how ', 'why ', 'explain ']):
            return 'explanation'
        
        return 'general'
    
    def adjust_thresholds_by_type(self, question_type: str, base_threshold: float) -> dict:
        """Return adjusted parameters based on question type.
        
        Args:
            question_type: The classified question type.
            base_threshold: The base QA score threshold to adjust from.
            
        Returns:
            Dictionary with adjusted parameters for this question type.
        """
        configs = {
            'factoid': {
                'qa_score_threshold': max(0.1, base_threshold),  # More lenient
                'max_answer_length': 50,     # Shorter answers
                'min_answer_words': 2,       # Allow shorter answers
                'prefer_entities': True
            },
            'explanation': {
                'qa_score_threshold': max(0.05, base_threshold * 0.5),  # Very lenient
                'max_answer_length': 200,    # Longer answers allowed
                'min_answer_words': 3,       # Normal minimum
                'prefer_entities': False
            },
            'comparison': {
                'qa_score_threshold': max(0.08, base_threshold * 0.8),
                'max_answer_length': 150,
                'min_answer_words': 3,
                'prefer_diversity': True     # Want different perspectives
            },
            'definition': {
                'qa_score_threshold': max(0.1, base_threshold),
                'max_answer_length': 100,
                'min_answer_words': 3,
                'prefer_entities': False
            },
            'general': {
                'qa_score_threshold': base_threshold,
                'max_answer_length': 150,
                'min_answer_words': 3,
                'prefer_entities': False
            }
        }
        return configs.get(question_type, configs['general'])
    
    def get_adaptive_context(self, paragraph: Paragraph, original_idx: int, 
                           all_paragraphs: List[Paragraph], 
                           question_type: str) -> Tuple[str, int, dict]:
        """Get context adaptively based on paragraph properties and question type.
        
        Args:
            paragraph: The current paragraph.
            original_idx: Index of the paragraph in all_paragraphs.
            all_paragraphs: Full list of all paragraphs.
            question_type: The classified question type.
            
        Returns:
            Tuple of (combined_text, shift_offset, context_info) where:
            - combined_text: The text to use for QA
            - shift_offset: Character position where current paragraph starts
            - context_info: Dict with context expansion details
        """
        combined_text = paragraph.text
        shift_offset = 0
        para_word_count = len(paragraph.text.split())
        
        context_info = {
            'expanded': False,
            'added_prev': False,
            'added_next': False,
            'original_words': para_word_count,
            'final_words': para_word_count,
            'expansion_reason': None
        }
        
        # Determine if context expansion is needed
        is_short = para_word_count < 50
        needs_more_context = question_type in ['explanation', 'comparison']
        
        should_expand = is_short or needs_more_context
        
        if is_short:
            context_info['expansion_reason'] = f'short paragraph ({para_word_count} words)'
        elif needs_more_context:
            context_info['expansion_reason'] = f'{question_type} question needs context'
        
        if not should_expand:
            return combined_text, shift_offset, context_info
        
        # Try to get previous paragraph if from same section and PDF
        if original_idx > 0:
            prev = all_paragraphs[original_idx - 1]
            if (prev.pdf_path == paragraph.pdf_path and 
                prev.section == paragraph.section):  # Same section
                combined_text = prev.text + " " + paragraph.text
                shift_offset = len(prev.text) + 1
                context_info['expanded'] = True
                context_info['added_prev'] = True
                context_info['final_words'] += len(prev.text.split())
        
        # For explanation questions, also try next paragraph
        if question_type == 'explanation' and original_idx < len(all_paragraphs) - 1:
            next_para = all_paragraphs[original_idx + 1]
            if (next_para.pdf_path == paragraph.pdf_path and 
                next_para.section == paragraph.section):
                combined_text = combined_text + " " + next_para.text
                context_info['expanded'] = True
                context_info['added_next'] = True
                context_info['final_words'] += len(next_para.text.split())
        
        return combined_text, shift_offset, context_info
    
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
        
        # Classify question type and adjust thresholds
        question_type = self.classify_question_type(question)
        config = self.adjust_thresholds_by_type(question_type, qa_score_threshold)
        logger.info(f"Question type: {question_type}, adjusted threshold: {config['qa_score_threshold']:.3f}")
        
        answers = []
        context_stats = {'expanded': 0, 'not_expanded': 0, 'added_prev': 0, 'added_next': 0}
        
        for i, (paragraph, retrieval_score, original_idx, rerank_score) in enumerate(candidates):
            if progress_callback:
                progress_callback(i, len(candidates), 
                                f"QA Analysis: Paragraph {i+1}/{len(candidates)}")
            
            # Get adaptive context based on question type
            combined_text, shift_offset, context_info = self.get_adaptive_context(
                paragraph, original_idx, all_paragraphs, question_type
            )
            
            # Track context expansion statistics
            if context_info['expanded']:
                context_stats['expanded'] += 1
                if context_info['added_prev']:
                    context_stats['added_prev'] += 1
                if context_info['added_next']:
                    context_stats['added_next'] += 1
            else:
                context_stats['not_expanded'] += 1
            
            # Run QA model
            qa_input = {'question': question, 'context': combined_text}
            result = self.pipeline(**qa_input)
            
            if result:
                # Filter based on question-type-specific minimum words
                if len(result['answer'].split()) < config['min_answer_words']:
                    continue
                
                raw_answer = result['answer']
                raw_start = result.get('start', 0)
                raw_end = result.get('end', len(raw_answer))
                
                # Determine which paragraph the answer belongs to
                # If shift_offset > 0, we have prepended context from previous paragraph(s)
                target_paragraph = paragraph
                local_start = raw_start
                local_end = raw_end
                
                if shift_offset > 0:
                    if raw_end <= shift_offset:
                        # Answer is entirely in previous context - try to find source paragraph
                        if original_idx > 0:
                            prev = all_paragraphs[original_idx - 1]
                            if prev.pdf_path == paragraph.pdf_path:
                                target_paragraph = prev
                    elif raw_start >= shift_offset:
                        # Answer is entirely in current paragraph
                        local_start = raw_start - shift_offset
                        local_end = raw_end - shift_offset
                    else:
                        # Answer spans contexts; use current paragraph
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
                    retrieval_score=retrieval_score,
                    rerank_score=rerank_score
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
            # Use question-type-specific threshold
            if ans.score < config['qa_score_threshold']:
                continue
                
            # Normalize text to catch minor variations
            norm_text = " ".join(ans.text.lower().split())
            signature = (ans.pdf_path, norm_text)
            
            if signature not in seen_answers:
                seen_answers.add(signature)
                unique_answers.append(ans)
        
        # Log context expansion statistics
        logger.info(f"Context expansion stats: {context_stats['expanded']} expanded "
                   f"({context_stats['added_prev']} +prev, {context_stats['added_next']} +next), "
                   f"{context_stats['not_expanded']} not expanded")
        logger.info(f"Extracted {len(unique_answers)} unique answers from {len(candidates)} candidates")
        
        return unique_answers
