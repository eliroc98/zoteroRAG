"""PDF folder source - alternative to Zotero database."""

import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class FolderPDFSource:
    """Handles PDFs from a folder instead of Zotero database."""
    
    def __init__(self, folder_path: str):
        """Initialize PDF folder source.
        
        Args:
            folder_path: Path to folder containing PDF files.
        """
        if not folder_path or not os.path.exists(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")
        
        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        self.folder_path = os.path.abspath(folder_path)
        logger.info(f"Initialized FolderPDFSource with folder: {self.folder_path}")
    
    def get_items(self, collection_name: str = None) -> List[Dict]:
        """Get PDF items from the folder.
        
        Args:
            collection_name: Ignored for folder source (for API compatibility).
            
        Returns:
            List of dictionaries with 'key', 'path', and 'title' keys.
        """
        items = []
        
        # Walk through the folder and find all PDFs
        for root, _dirs, files in os.walk(self.folder_path):
            for filename in files:
                if filename.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, filename)
                    
                    # Use filename (without extension) as title
                    title = os.path.splitext(filename)[0]
                    
                    # Use relative path as key (for uniqueness)
                    rel_path = os.path.relpath(pdf_path, self.folder_path)
                    key = rel_path.replace(os.sep, '_')
                    
                    items.append({
                        'key': key,
                        'path': pdf_path,
                        'title': title
                    })
        
        logger.info(f"Found {len(items)} PDF files in {self.folder_path}")
        return items
    
    def list_collections(self) -> List[Dict]:
        """Return empty list for API compatibility with ZoteroDatabase.
        
        Returns:
            Empty list (folders don't have collections).
        """
        return []
