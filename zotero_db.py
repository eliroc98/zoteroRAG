"""Zotero database interaction utilities."""

import os
import sqlite3
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ZoteroDatabase:
    """Handles interaction with the Zotero SQLite database."""
    
    @staticmethod
    def find_zotero_dir(custom_dir: str = None) -> str:
        """Find Zotero data directory."""
        if custom_dir and os.path.exists(custom_dir):
            return custom_dir
        possible_paths = [
            os.path.expanduser(p) 
            for p in ["~/Zotero", "~/Documents/Zotero", "~/.zotero"]
        ]
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'zotero.sqlite')):
                return path
        raise ValueError("Zotero directory not found. Please specify zotero_data_dir")
    
    def __init__(self, zotero_data_dir: str = None):
        """Initialize Zotero database connection.
        
        Args:
            zotero_data_dir: Path to Zotero data directory. If None, auto-detect.
        """
        self.zotero_dir = self.find_zotero_dir(zotero_data_dir)
        self.storage_dir = os.path.join(self.zotero_dir, 'storage')
        self.db_path = os.path.join(self.zotero_dir, 'zotero.sqlite')
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Zotero database not found at {self.db_path}")
    
    def list_collections(self) -> List[Dict]:
        """Load collections from the Zotero database.
        
        Returns:
            List of dictionaries with 'id', 'name', and 'parent_id' keys.
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT collectionID, collectionName, parentCollectionID "
            "FROM collections ORDER BY collectionName"
        )
        collections = [
            {'id': r[0], 'name': r[1], 'parent_id': r[2]} 
            for r in cursor.fetchall()
        ]
        conn.close()
        return collections
    
    def get_items(self, collection_name: str = None) -> List[Dict]:
        """Get PDF items from Zotero library or a specific collection.
        
        Args:
            collection_name: Name of the collection to filter by. If None, get all items.
            
        Returns:
            List of dictionaries with 'key', 'path', and 'title' keys.
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()
        items = []
        
        if collection_name:
            # Get items from specific collection
            cursor.execute(
                "SELECT collectionID FROM collections WHERE collectionName = ?", 
                (collection_name,)
            )
            result = cursor.fetchone()
            if not result:
                conn.close()
                raise ValueError(f"Collection '{collection_name}' not found.")
            collection_id = result[0]
            
            query = """
            SELECT a_items.key, ia.path, ia.parentItemID as sourceItemID 
            FROM collectionItems ci
            JOIN itemAttachments ia ON ci.itemID = ia.parentItemID 
            JOIN items a_items ON ia.itemID = a_items.itemID
            WHERE ci.collectionID = ? 
            AND ia.contentType = 'application/pdf' 
            AND ia.path IS NOT NULL
            """
            cursor.execute(query, (collection_id,))
        else:
            # Get all PDF items from library
            query = """
            SELECT i.key, ia.path, COALESCE(ia.parentItemID, i.itemID) as sourceItemID
            FROM items i 
            JOIN itemAttachments ia ON i.itemID = ia.itemID
            WHERE ia.contentType = 'application/pdf' 
            AND ia.path IS NOT NULL
            """
            cursor.execute(query)
        
        rows = cursor.fetchall()
        for key, path, src_id in rows:
            # Get title from parent item
            cursor.execute(
                "SELECT v.value FROM itemData d "
                "JOIN itemDataValues v ON d.valueID = v.valueID "
                "JOIN fields f ON d.fieldID = f.fieldID "
                "WHERE d.itemID = ? AND f.fieldName = 'title'", 
                (src_id,)
            )
            title = (r[0] if (r := cursor.fetchone()) else "Unknown")
            
            # Resolve storage path
            if path and path.startswith('storage:'):
                pdf_path = os.path.join(
                    self.storage_dir, 
                    key, 
                    path.replace('storage:', '')
                )
            else:
                pdf_path = path
            
            # Only include if file exists
            if pdf_path and os.path.exists(pdf_path):
                items.append({
                    'key': key, 
                    'path': pdf_path, 
                    'title': title
                })
        
        conn.close()
        return items
