"""Data models for the Zotero RAG system."""

from dataclasses import dataclass, field
from typing import List, Tuple


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
    rerank_score: float = 0.0  # CrossEncoder reranking score
    
    def __reduce__(self):
        """Custom pickle support for dataclass."""
        return (
            self.__class__,
            (self.text, self.context, self.pdf_path, self.page_num, self.item_key, self.title,
             self.section, self.start_char, self.end_char, self.score, self.query, self.color, 
             self.sentence_coords, self.retrieval_score, self.rerank_score)
        )
