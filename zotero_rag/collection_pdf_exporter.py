"""Export PDFs from Zotero collections maintaining folder structure."""

import os
import shutil
import sqlite3
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class CollectionPDFExporter:
    """Export PDFs from a Zotero collection with subcollections to a folder structure."""
    
    def __init__(self, zotero_data_dir: Optional[str] = None):
        """Initialize the exporter.
        
        Args:
            zotero_data_dir: Path to Zotero data directory. If None, auto-detect.
        """
        self.zotero_dir = self._find_zotero_dir(zotero_data_dir)
        self.storage_dir = os.path.join(self.zotero_dir, 'storage')
        self.db_path = os.path.join(self.zotero_dir, 'zotero.sqlite')
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Zotero database not found at {self.db_path}")
        
        logger.info(f"Using Zotero database at: {self.db_path}")
    
    @staticmethod
    def _find_zotero_dir(custom_dir: Optional[str] = None) -> str:
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
    
    def _get_collection_hierarchy(self, collection_name: str) -> Dict:
        """Get collection and all its subcollections in hierarchical structure.
        
        Args:
            collection_name: Name of the root collection.
            
        Returns:
            Dictionary representing collection hierarchy.
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()
        
        # Find root collection
        cursor.execute(
            "SELECT collectionID, collectionName FROM collections WHERE collectionName = ?",
            (collection_name,)
        )
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            raise ValueError(f"Collection '{collection_name}' not found")
        
        root_id, root_name = result
        
        # Get all collections with their parent relationships
        cursor.execute(
            "SELECT collectionID, collectionName, parentCollectionID FROM collections"
        )
        all_collections = {row[0]: {'name': row[1], 'parent_id': row[2]} for row in cursor.fetchall()}
        conn.close()
        
        # Build hierarchy starting from root
        def build_tree(coll_id: int, path: List[str]) -> Dict:
            """Recursively build collection tree."""
            current_name = all_collections[coll_id]['name']
            current_path = path + [current_name]
            
            # Find children
            children = [
                build_tree(child_id, current_path)
                for child_id, info in all_collections.items()
                if info['parent_id'] == coll_id
            ]
            
            return {
                'id': coll_id,
                'name': current_name,
                'path': current_path,
                'children': children
            }
        
        return build_tree(root_id, [])
    
    def _get_pdfs_in_collection(self, collection_id: int) -> List[Dict]:
        """Get all PDF attachments in a specific collection.
        
        Args:
            collection_id: ID of the collection.
            
        Returns:
            List of dictionaries with PDF information.
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()
        
        # Get PDFs in this collection
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
        
        items = []
        for key, path, src_id in cursor.fetchall():
            # Get title from parent item
            cursor.execute(
                "SELECT v.value FROM itemData d "
                "JOIN itemDataValues v ON d.valueID = v.valueID "
                "JOIN fields f ON d.fieldID = f.fieldID "
                "WHERE d.itemID = ? AND f.fieldName = 'title'",
                (src_id,)
            )
            title_result = cursor.fetchone()
            title = title_result[0] if title_result else "Unknown"
            
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
    
    def _export_collection_recursive(
        self, 
        collection_info: Dict, 
        output_base: Path,
        stats: Dict
    ) -> None:
        """Recursively export PDFs from collection and subcollections.
        
        Args:
            collection_info: Collection hierarchy dictionary.
            output_base: Base output directory path.
            stats: Dictionary to track export statistics.
        """
        # Create folder for this collection
        # Use only the collection name (not full path) for the folder
        collection_folder = output_base / collection_info['name']
        collection_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing collection: {' / '.join(collection_info['path'])}")
        
        # Get PDFs in this collection
        pdfs = self._get_pdfs_in_collection(collection_info['id'])
        
        # Export PDFs
        for pdf in pdfs:
            # Sanitize filename
            safe_title = self._sanitize_filename(pdf['title'])
            dest_filename = f"{safe_title}.pdf"
            dest_path = collection_folder / dest_filename
            
            # Handle duplicate filenames
            counter = 1
            while dest_path.exists():
                dest_filename = f"{safe_title}_{counter}.pdf"
                dest_path = collection_folder / dest_filename
                counter += 1
            
            try:
                shutil.copy2(pdf['path'], dest_path)
                logger.info(f"  Copied: {pdf['title']} -> {dest_path.name}")
                stats['copied'] += 1
            except Exception as e:
                logger.error(f"  Failed to copy {pdf['title']}: {e}")
                stats['failed'] += 1
        
        # Process subcollections
        for child in collection_info['children']:
            self._export_collection_recursive(child, collection_folder, stats)
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename for filesystem compatibility.
        
        Args:
            filename: Original filename.
            
        Returns:
            Sanitized filename.
        """
        # Replace problematic characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Trim to reasonable length
        max_length = 200
        if len(filename) > max_length:
            filename = filename[:max_length]
        
        return filename.strip()
    
    def export_collection(
        self, 
        collection_name: str, 
        output_dir: str,
        include_root_folder: bool = True
    ) -> Dict:
        """Export all PDFs from a collection and its subcollections.
        
        Args:
            collection_name: Name of the collection to export.
            output_dir: Directory where PDFs should be exported.
            include_root_folder: If True, creates a folder for the root collection.
                                If False, exports directly to output_dir.
            
        Returns:
            Dictionary with export statistics.
        """
        # Get collection hierarchy
        collection_tree = self._get_collection_hierarchy(collection_name)
        
        # Prepare output directory
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine base path for export
        if include_root_folder:
            base_path = output_path
        else:
            # Export directly to output_dir without creating root collection folder
            base_path = output_path.parent
            collection_tree['name'] = output_path.name
        
        # Statistics
        stats = {'copied': 0, 'failed': 0}
        
        # Start recursive export
        self._export_collection_recursive(collection_tree, base_path, stats)
        
        logger.info(f"\nExport complete!")
        logger.info(f"  PDFs copied: {stats['copied']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Output directory: {output_path}")
        
        return stats


def main():
    """Command-line interface for exporting PDFs."""
    parser = argparse.ArgumentParser(
        description="Export PDFs from a Zotero collection maintaining folder structure"
    )
    parser.add_argument(
        'collection_name',
        help='Name of the Zotero collection to export'
    )
    parser.add_argument(
        'output_dir',
        help='Directory where PDFs should be exported'
    )
    parser.add_argument(
        '--zotero-dir',
        help='Path to Zotero data directory (auto-detect if not specified)'
    )
    parser.add_argument(
        '--no-root-folder',
        action='store_true',
        help='Export directly to output_dir without creating root collection folder'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create exporter and run
        exporter = CollectionPDFExporter(zotero_data_dir=args.zotero_dir)
        exporter.export_collection(
            collection_name=args.collection_name,
            output_dir=args.output_dir,
            include_root_folder=not args.no_root_folder
        )
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


if __name__ == '__main__':
    main()
