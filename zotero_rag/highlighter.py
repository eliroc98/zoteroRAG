"""PDF highlighting utilities using TEI coordinates."""

import os
import logging
import tempfile
import shutil
from typing import List

from models import Answer

logger = logging.getLogger(__name__)


class PDFHighlighter:
    """Handles highlighting of PDFs using TEI sentence coordinates."""
    
    def __init__(self):
        """Initialize the PDF highlighter."""
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
            self.available = True
        except ImportError:
            logger.warning("PyMuPDF (fitz) not available for highlighting")
            self.fitz = None
            self.available = False
    
    def highlight_pdf(self, answers: List[Answer], output_path: str) -> str:
        """Highlight PDF using TEI sentence coordinates for precise highlighting.
        
        Args:
            answers: List of Answer objects to highlight (must be from same PDF).
            output_path: Path where the highlighted PDF should be saved.
            
        Returns:
            Path to the highlighted PDF, or None if highlighting failed.
        """
        if not self.available or not answers:
            return None
        
        try:
            # Use previously highlighted PDF if it exists (to preserve previous highlights),
            # otherwise use original PDF
            source_pdf = output_path if os.path.exists(output_path) else answers[0].pdf_path
            doc = self.fitz.open(source_pdf)
            
            # Group answers by the actual pages their coordinates reference
            # (not by answer.page_num which may be wrong for multi-page paragraphs)
            coords_by_page = {}
            for answer in answers:
                if answer.sentence_coords:
                    for coords_str in answer.sentence_coords:
                        for coord_group in coords_str.split(';'):
                            try:
                                parts = coord_group.strip().split(',')
                                if len(parts) >= 5:
                                    # GROBID uses 1-indexed pages
                                    coord_page = int(parts[0]) - 1
                                    if coord_page not in coords_by_page:
                                        coords_by_page[coord_page] = []
                                    coords_by_page[coord_page].append((answer, coord_group))
                            except (ValueError, IndexError):
                                continue
            
            # Process each page that has coordinates
            for page_num, page_data in coords_by_page.items():
                if page_num >= len(doc):
                    logger.warning(f"Page {page_num} referenced in coords but PDF only has {len(doc)} pages")
                    continue
                    
                page = doc[page_num]
                
                # Group by answer ID to track if we highlighted anything for each
                # (using id() since Answer objects aren't hashable)
                answer_highlighted = {}
                answer_by_id = {}
                
                for answer, coord_group in page_data:
                    answer_id = id(answer)
                    if answer_id not in answer_highlighted:
                        answer_highlighted[answer_id] = False
                        answer_by_id[answer_id] = answer
                        
                    try:
                        parts = coord_group.strip().split(',')
                        if len(parts) >= 5:
                            x0 = float(parts[1])
                            y0 = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            # Convert to (x0, y0, x1, y1) format for PyMuPDF
                            x1 = x0 + width
                            y1 = y0 + height
                            # Create rectangle for this text region
                            rect = self.fitz.Rect(x0, y0, x1, y1)
                            highlight = page.add_highlight_annot(rect)
                            highlight.set_colors(stroke=answer.color)
                            highlight.update()
                            answer_highlighted[answer_id] = True
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Could not parse coordinates '{coord_group}': {e}")
                        continue
                
                # Add annotations for answers that were highlighted on this page
                for answer_id, was_highlighted in answer_highlighted.items():
                    if was_highlighted:
                        answer = answer_by_id[answer_id]
                        if answer.sentence_coords:
                            try:
                                # Find first coord on this page for annotation placement
                                found_annotation_spot = False
                                for coords_str in answer.sentence_coords:
                                    for coord_group in coords_str.split(';'):
                                        parts = coord_group.strip().split(',')
                                        if len(parts) >= 5:
                                            coord_page = int(parts[0]) - 1
                                            if coord_page == page_num:
                                                x0, y0 = float(parts[1]), float(parts[2])
                                                answer_preview = answer.text[:100] + "..." \
                                                                if len(answer.text) > 100 else answer.text
                                                annot_text = (
                                                    f"Q: {answer.query}\n\n"
                                                    f"Scores:\n"
                                                    f"• Retrieval: {answer.retrieval_score:.4f}\n"
                                                    f"• Rerank: {answer.rerank_score:.4f}\n"
                                                    f"• QA Confidence: {answer.score:.4f}"
                                                )
                                                annot = page.add_text_annot(
                                                    self.fitz.Point(x0, y0),
                                                    annot_text
                                                )
                                                annot.update()
                                                found_annotation_spot = True
                                                break
                                    if found_annotation_spot:
                                        break
                            except Exception as e:
                                logger.debug(f"Could not annotate on page {page_num}: {e}")
                                continue
            
            # Save the PDF with all highlights
            # Always create a temporary file first, then replace the original
            # This avoids incremental save issues and encryption problems
            temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf', dir=os.path.dirname(output_path))
            os.close(temp_fd)  # Close the file descriptor, we'll use the path
            
            try:
                # Save to temporary file with non-incremental save
                try:
                    doc.save(temp_path, incremental=False, deflate=True)
                except Exception as e:
                    # Fallback without deflate if it fails
                    error_msg = str(e).lower()
                    if "deflate" in error_msg or "encryption" in error_msg:
                        logger.debug(f"Save with deflate failed, retrying without: {e}")
                        doc.save(temp_path, incremental=False, deflate=False)
                    else:
                        raise
                
                doc.close()
                
                # Replace the original file with the new one
                shutil.move(temp_path, output_path)
                
            except Exception as e:
                # Clean up temp file if something went wrong
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise
            
            logger.info(f"PDF highlighted and saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error highlighting PDF: {e}", exc_info=True)
            return None
