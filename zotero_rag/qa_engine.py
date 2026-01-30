"""Question answering engine using extractive QA models."""

import logging
from typing import List, Tuple, Optional
from transformers import pipeline
import numpy as np

from models import Paragraph, Answer

logger = logging.getLogger(__name__)


class QAEngine:
    """Handles extractive question answering over text passages."""
    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2", 
                 device: str = None,
                 enable_question_expansion: bool = True):
        """Initialize the QA engine.
        
        Args:
            model_name: Name of the HuggingFace QA model.
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detect if None.
            enable_question_expansion: Generate question variations to improve retrieval.
        """
        import torch
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.enable_question_expansion = enable_question_expansion
        self.pipeline = None
        self.paraphraser = None
        self._load_pipeline()
        if enable_question_expansion:
            self._load_paraphraser()
    
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
    
    def _load_paraphraser(self):
        """Load paraphrasing model for question expansion."""
        try:
            # Use a lightweight paraphrasing model
            # Force CPU for stability on MPS
            self.paraphraser = pipeline(
                "text2text-generation",
                model="humarin/chatgpt_paraphraser_on_T5_base",
                device=-1  # Always use CPU (more stable across platforms)
            )
            logger.info("Question paraphraser loaded: humarin/chatgpt_paraphraser_on_T5_base (CPU)")
        except Exception as e:
            logger.warning(f"Could not load paraphraser: {e}. Question expansion disabled.")
            self.paraphraser = None
    
    def get_config_for_type(self, question_type: str, base_threshold: float) -> dict:
        """Return configuration for a specific question type.
        
        Args:
            question_type: The question type ('factoid', 'methodology', 'explanation', etc.).
            base_threshold: The base QA score threshold to use/adjust from.
            
        Returns:
            Dictionary with parameters for this question type.
        """
        configs = {
            'factoid': {
                'qa_score_threshold': max(0.1, base_threshold),  # More lenient
                'max_answer_length': 50,     # Shorter answers
                'min_answer_words': 2,       # Allow shorter answers
                'prefer_entities': True
            },
            'methodology': {
                'qa_score_threshold': max(0.05, base_threshold * 0.5),  # Very lenient
                'max_answer_length': 250,    # Longer answers for detailed explanations
                'min_answer_words': 5,       # Ensure substantial answers
                'prefer_entities': False,
                'section_diversity': True,   # Want both high-level and detailed answers
                'priority_sections': ['abstract', 'introduction', 'methodology', 'methods', 
                                     'approach', 'algorithm', 'implementation']
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
            },
            'custom': {
                # Custom will be overridden by provided config
                'qa_score_threshold': base_threshold,
                'max_answer_length': 150,
                'min_answer_words': 3,
                'prefer_entities': False
            }
        }
        return configs.get(question_type, configs['general'])
    
    def expand_question(self, question: str, num_variations: int = 2) -> List[str]:
        """Generate question variations to improve retrieval coverage.
        
        Args:
            question: The original question.
            num_variations: Number of paraphrases to generate.
            
        Returns:
            List of question variations including the original.
        """
        variations = [question]  # Always include original
        
        if not self.paraphraser or not self.enable_question_expansion:
            return variations
        
        try:
            # Generate multiple candidates with higher diversity
            all_candidates = []
            
            # Generate more candidates than needed to filter best ones
            result = self.paraphraser(
                f"paraphrase: {question}",  # Add explicit instruction
                max_new_tokens=64,  # Use max_new_tokens instead of max_length
                num_return_sequences=num_variations * 3,  # Generate 3x more for filtering
                num_beams=num_variations * 3,
                temperature=1.5,  # Higher temperature for more diversity
                top_k=50,
                top_p=0.95,
                do_sample=True,
                early_stopping=True
            )
            
            # Collect and filter paraphrases
            for res in result:
                paraphrase = res['generated_text'].strip()
                
                # Remove common prefixes that might be added
                paraphrase = paraphrase.replace('paraphrase:', '').strip()
                
                # Check if meaningfully different from original
                if paraphrase and paraphrase != question:
                    normalized = paraphrase.lower().strip()
                    if normalized != question.lower():
                        # Calculate similarity to original
                        orig_words = set(question.lower().split())
                        new_words = set(paraphrase.lower().split())
                        
                        intersection = len(orig_words & new_words)
                        union = len(orig_words | new_words)
                        similarity = intersection / union if union > 0 else 1.0
                        
                        # Keep if diverse enough from original (< 75% similar)
                        if similarity < 0.75:
                            all_candidates.append((similarity, paraphrase))
            
            # Sort by diversity from original (lower similarity = more diverse)
            all_candidates.sort(key=lambda x: x[0])
            
            # Select candidates while checking diversity among variations themselves
            for similarity, candidate in all_candidates:
                if len(variations) - 1 >= num_variations:
                    break
                
                # Check candidate is diverse from already-selected variations
                is_diverse = True
                candidate_words = set(candidate.lower().split())
                
                for existing_var in variations:
                    existing_words = set(existing_var.lower().split())
                    var_intersection = len(candidate_words & existing_words)
                    var_union = len(candidate_words | existing_words)
                    var_similarity = var_intersection / var_union if var_union > 0 else 1.0
                    
                    # Require < 70% similarity to all existing variations
                    if var_similarity >= 0.70:
                        is_diverse = False
                        break
                
                if is_diverse:
                    variations.append(candidate)
            
            logger.info(f"Generated {len(variations)-1} question variations from {len(all_candidates)} candidates")
            for i, var in enumerate(variations):
                logger.debug(f"  Q{i}: {var}")
                
        except Exception as e:
            logger.warning(f"Question expansion failed: {e}")
        
        return variations
    
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
                       progress_callback=None,
                       question_variations: List[str] = None,
                       question_type: str = 'general',
                       custom_config: dict = None) -> List[Answer]:
        """Extract answers from candidate paragraphs using QA model.
        
        Args:
            question: The question to answer.
            candidates: List of (Paragraph, retrieval_score, original_index, rerank_score) tuples.
            all_paragraphs: Full list of all paragraphs (for context overlap).
            qa_score_threshold: Minimum QA confidence score threshold.
            color: RGB color tuple for highlighting.
            progress_callback: Function(current, total, message) for progress updates.
            question_variations: List of question paraphrases to average scores over.
            question_type: The type of question (factoid, explanation, etc.).
            custom_config: Custom configuration dict to override preset config.
            
        Returns:
            List of Answer objects, deduplicated and sorted by score.
        """
        if not self.pipeline:
            raise RuntimeError("QA pipeline not available; cannot answer question")
        
        if not candidates:
            return []
        
        # Use question variations if provided, otherwise just use the original question
        questions_to_use = question_variations if question_variations else [question]
        
        # Get configuration for the specified question type
        config = self.get_config_for_type(question_type, qa_score_threshold)
        
        # Override with custom config if provided
        if custom_config:
            config.update(custom_config)
        
        logger.info(f"Question type: {question_type}, QA threshold: {config['qa_score_threshold']:.3f}")
        
        answers = []
        context_stats = {'expanded': 0, 'not_expanded': 0, 'added_prev': 0, 'added_next': 0}
        filter_stats = {'total_processed': 0, 'too_few_words': 0, 'below_threshold': 0, 'added': 0}
        
        total_operations = len(candidates) * len(questions_to_use)
        completed_operations = 0
        
        for i, (paragraph, retrieval_score, original_idx, rerank_score) in enumerate(candidates):
            filter_stats['total_processed'] += 1
            
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
            
            # Run QA model with each question variation and average scores
            all_results = []
            for var_idx, q_var in enumerate(questions_to_use):
                qa_input = {'question': q_var, 'context': combined_text}
                var_result = self.pipeline(**qa_input)
                if var_result:
                    all_results.append(var_result)
                
                completed_operations += 1
                
                # Update progress with paraphrase info
                if progress_callback:
                    variation_info = f"Paraphrase {var_idx + 1}/{len(questions_to_use)}" if len(questions_to_use) > 1 else ""
                    para_info = f"Paragraph {i+1}/{len(candidates)}"
                    message = f"{variation_info} - {para_info}" if variation_info else para_info
                    progress_callback(completed_operations, total_operations, message)
            
            
            # Use max score across variations, with the answer from the highest scoring variation
            best_result = max(all_results, key=lambda r: r.get('score', 0.0))
            max_score = best_result.get('score', 0.0)
            result = best_result.copy()
            result['score'] = max_score
            
            if len(questions_to_use) > 1:
                logger.debug(f"Using max QA score across {len(questions_to_use)} variations: {max_score:.3f}")
            
            # Filter based on question-type-specific minimum words
            answer_word_count = len(result['answer'].split())
            if answer_word_count < config['min_answer_words']:
                logger.debug(f"Filtered answer (too few words: {answer_word_count} < {config['min_answer_words']}): '{result['answer'][:50]}...'")
                filter_stats['too_few_words'] += 1
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
            filter_stats['added'] += 1
            answers.append(answer)
        
        if progress_callback:
            progress_callback(len(candidates), len(candidates), "Finalizing results...")
        
        # Sort by QA score descending
        answers.sort(key=lambda x: x.score, reverse=True)
        
        # --- DEDUPLICATION ---
        unique_answers = []
        
        if config.get('section_diversity', False):
            # For methodology questions: ensure section diversity
            # Group answers by section type (intro/abstract vs detailed methodology)
            intro_sections = ['abstract', 'introduction', 'intro']
            method_sections = ['methodology', 'methods', 'approach', 'algorithm', 
                             'implementation', 'procedure', 'technique']
            
            intro_answers = []
            method_answers = []
            other_answers = []
            
            for ans in answers:
                if ans.score < config['qa_score_threshold']:
                    logger.debug(f"Filtered answer (score {ans.score:.3f} < {config['qa_score_threshold']:.3f}): '{ans.text[:50]}...'")
                    filter_stats['below_threshold'] += 1
                    continue
                    
                section_lower = (ans.section or '').lower()
                
                # Categorize by section type
                if any(s in section_lower for s in intro_sections):
                    intro_answers.append(ans)
                elif any(s in section_lower for s in method_sections):
                    method_answers.append(ans)
                else:
                    other_answers.append(ans)
            
            # Deduplicate within each group
            def deduplicate_group(group):
                seen = set()
                unique = []
                for ans in group:
                    norm_text = " ".join(ans.text.lower().split())
                    signature = (ans.pdf_path, norm_text)
                    if signature not in seen:
                        seen.add(signature)
                        unique.append(ans)
                return unique
            
            intro_unique = deduplicate_group(intro_answers)
            method_unique = deduplicate_group(method_answers)
            other_unique = deduplicate_group(other_answers)
            
            # Interleave intro and method answers to show both perspectives
            max_len = max(len(intro_unique), len(method_unique))
            for i in range(max_len):
                if i < len(intro_unique):
                    unique_answers.append(intro_unique[i])
                if i < len(method_unique):
                    unique_answers.append(method_unique[i])
            unique_answers.extend(other_unique)
            
            logger.info(f"Section diversity: {len(intro_unique)} intro/abstract, "
                       f"{len(method_unique)} methodology, {len(other_unique)} other")
        else:
            # Standard deduplication based on (pdf_path, normalized_text)
            seen_answers = set()
            
            for ans in answers:
                # Use question-type-specific threshold
                if ans.score < config['qa_score_threshold']:
                    logger.debug(f"Filtered answer (score {ans.score:.3f} < {config['qa_score_threshold']:.3f}): '{ans.text[:50]}...'")
                    filter_stats['below_threshold'] += 1
                    continue
                    
                # Normalize text to catch minor variations
                norm_text = " ".join(ans.text.lower().split())
                signature = (ans.pdf_path, norm_text)
                
                if signature not in seen_answers:
                    seen_answers.add(signature)
                    unique_answers.append(ans)
        
        # Log comprehensive statistics
        logger.info(f"Context expansion stats: {context_stats['expanded']} expanded "
                   f"({context_stats['added_prev']} +prev, {context_stats['added_next']} +next), "
                   f"{context_stats['not_expanded']} not expanded")
        logger.info(f"Filter stats: {filter_stats['total_processed']} processed -> "
                   f"{filter_stats['added']} passed word filter -> "
                   f"{len(answers)} pre-dedup -> "
                   f"{len(unique_answers)} final unique answers")
        if filter_stats['too_few_words'] > 0:
            logger.info(f"  Filtered out: {filter_stats['too_few_words']} (too few words), "
                       f"{filter_stats['below_threshold']} (below QA threshold)")
        
        return unique_answers
