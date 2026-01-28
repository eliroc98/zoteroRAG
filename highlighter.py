"""PDF highlighting utilities using TEI coordinates."""

import os
import logging
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
            
            # Group answers by page
            answers_by_page = {}
            for answer in answers:
                answers_by_page.setdefault(answer.page_num, []).append(answer)
            
            # Process each page
            for page_num, page_answers in answers_by_page.items():
                page = doc[page_num]
                
                for answer in page_answers:
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
                                            rect = self.fitz.Rect(x0, y0, x1, y1)
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
                                    answer_preview = answer.text[:100] + "..." \
                                                    if len(answer.text) > 100 else answer.text
                                    annot = page.add_text_annot(
                                        self.fitz.Point(x0, y0),
                                        f"Q: {answer.query[:80]}...\nA: {answer_preview}"
                                    )
                                    annot.update()
                            except:
                                logger.warning(f"Could not annotate {answer.pdf_path}.")
                                continue
                        else:
                            logger.warning(
                                f"Could not highlight {answer.pdf_path} using coordinates "
                                f"on page {page_num}."
                            )

                    if not highlighted_any:
                        logger.warning(
                            f"Could not highlight {answer.pdf_path} using TEI coordinates "
                            f"on page {page_num}: {answer.text[:50]}... Skipping fallback search."
                        )
            
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
            
            logger.info(f"PDF highlighted and saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error highlighting PDF: {e}", exc_info=True)
            return None
