"""
app.py - Streamlit web interface for Zotero RAG System

Run with: streamlit run app.py
"""

import os
import sys
from typing import List, Dict, Tuple
import subprocess
import sqlite3
import streamlit as st
from zotero_rag import ZoteroRAG
import re

def _sanitize_filename(name: str) -> str:
    """Converts a string into a safe filename."""
    if not name:
        return "_All_Library"
    s = name.replace(" ", "_")
    s = re.sub(r'(?u)[^-\w.]', '', s)
    return s

def _sanitize_model_name(model_name: str) -> str:
    """Convert model name to safe filename component."""
    model_short = model_name.split('/')[-1]
    return re.sub(r'[^a-zA-Z0-9_-]', '_', model_short)

def find_saved_indexes(collection_name: str, model_name: str = None, base_dir: str = "output") -> List[str]:
    """Finds all saved index files within the collection's dedicated folder."""
    collection_folder_name = _sanitize_filename(collection_name)
    index_dir = os.path.join(base_dir, collection_folder_name)

    if not os.path.isdir(index_dir):
        return []

    found_files = []
    model_suffix = f"_{_sanitize_model_name(model_name)}" if model_name else ""
    
    for f in os.listdir(index_dir):
        if f.endswith('.index'):
            # If model specified, only return indexes for that model
            if model_name and model_suffix not in f:
                continue
            base_name = os.path.splitext(f)[0]
            full_path_base = os.path.join(index_dir, base_name)
            found_files.append(full_path_base)
    
    return sorted([os.path.basename(f) for f in found_files])

def rgb_to_hex(rgb):
    """Convert RGB tuple (0-1) to hex color"""
    r, g, b = [int(x * 255) for x in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'


def main():
    st.set_page_config(
        page_title="Zotero RAG Navigator",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Zotero RAG Navigator")
    st.markdown("Query your local Zotero library with natural language")

    # Output directory (first thing asked)
    st.subheader("🗂️ Output Directory")
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = "./output"
    output_dir = st.text_input(
        "Base output directory (indexes, TEI cache, highlights)",
        value=st.session_state.output_dir,
        help="Choose where to store indexes, cached TEI files, and highlighted PDFs."
    )
    st.session_state.output_dir = output_dir or "./output"
    
    # Initialize session state
    if 'rag' not in st.session_state:
        st.session_state.rag = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'indexed' not in st.session_state:
        st.session_state.indexed = False
    if 'collections_loaded' not in st.session_state:
        st.session_state.collections_loaded = False
    if 'collections' not in st.session_state:
        st.session_state.collections = []
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = None
    if 'model_name' not in st.session_state:
        st.session_state.model_name = "BAAI/bge-small-en-v1.5"
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model_device' not in st.session_state:
        st.session_state.model_device = None  # auto-select
    
    # Load collections on first run
    if not st.session_state.collections_loaded:
        try:
            with st.spinner("Loading Zotero collections..."):
                st.session_state.collections = ZoteroRAG.list_collections()
                st.session_state.collections_loaded = True
        except sqlite3.OperationalError as e:
            if "locked" in str(e):
                st.error("⚠️ Zotero database is locked")
                st.warning("""
                **The database is currently locked by Zotero.**
                
                You have two options:
                
                1. **Close Zotero** (Recommended)
                   - Close the Zotero application completely
                   - Then refresh this page
                
                2. **Keep Zotero open** (Advanced)
                   - The app will try to read the database in read-only mode
                   - Click the button below to retry
                """)
                
                if st.button("🔄 Retry Connection"):
                    st.rerun()
            else:
                st.error(f"Database error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Error loading Zotero: {e}")
            st.info("Make sure Zotero is installed and the database is accessible")
            st.stop()
    
    # Main tabs - only show after model is loaded and indexed
    if st.session_state.model_loaded and st.session_state.indexed:
        tab1, tab2 = st.tabs(["⚙️ Setup", "🔍 Search & Highlight"])
        
        with tab1:
            show_setup_tab()
        
        with tab2:
            show_search_tab()
    else:
        # Show setup only
        show_setup_tab()


def show_setup_tab():
    """Setup tab for collection selection, model loading, and indexing."""
    
    st.header("⚙️ Setup Configuration")
    
    # Collection Selection
    st.subheader("1️⃣ Select Collection")
    collection_options = ["All Library"]
    for coll in st.session_state.collections:
        name = coll['name']
        if coll['parent_id']:
            parent_name = next((c['name'] for c in st.session_state.collections 
                              if c['id'] == coll['parent_id']), "Unknown")
            name = f"{parent_name} > {name}"
        collection_options.append(name)
    
    selected_collection = st.selectbox(
        "Choose which Zotero collection to search",
        collection_options,
        key="collection_selector"
    )
    
    st.session_state.collection_name = None if selected_collection == "All Library" else selected_collection.split(" > ")[-1].strip()
    
    st.markdown("---")
    
    # GROBID Configuration
    st.subheader("2️⃣ GROBID Service (Optional)")
    st.info("🔧 GROBID is used for advanced PDF parsing with sentence-level extraction. Leave default if not running.")
    
    grobid_url = st.text_input(
        "GROBID Service URL",
        value="http://localhost:8070",
        help="URL of GROBID service. Start with: docker run -p 8070:8070 grobid/grobid:latest"
    )
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("3️⃣ Select Embedding Model")
    
    model_input = st.text_input(
        "HuggingFace Model URL or name",
        value=st.session_state.model_name,
        placeholder="e.g., sentence-transformers/all-MiniLM-L6-v2",
        help="Enter the HuggingFace model identifier (e.g., 'all-MiniLM-L6-v2' or 'BAAI/bge-small-en-v1.5')"
    )

    device_choice = st.selectbox(
        "Device",
        options=["auto", "cpu", "mps", "cuda"],
        index=["auto", "cpu", "mps", "cuda"].index(
            "auto" if st.session_state.model_device is None else st.session_state.model_device
        ),
        help="Select compute device. 'auto' picks MPS if available, else CPU."
    )
    
    if st.button("📥 Load Model", type="primary", use_container_width=True):
        if model_input:
            with st.spinner(f"Loading model: {model_input}..."):
                try:
                    st.session_state.model_name = model_input
                    st.session_state.grobid_url = grobid_url
                    st.session_state.model_device = None if device_choice == "auto" else device_choice
                    st.session_state.rag = ZoteroRAG(
                        collection_name=st.session_state.collection_name,
                        model_name=model_input,
                        grobid_url=grobid_url,
                        output_base_dir=st.session_state.output_dir,
                        model_device=st.session_state.model_device
                    )
                    st.session_state.model_loaded = True
                    st.session_state.indexed = False  # Reset indexed status
                    st.success(f"✅ Model loaded: {model_input}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    st.info("Make sure the model name is correct and available on HuggingFace")
        else:
            st.error("Please enter a model name")
    
    if st.session_state.model_loaded:
        st.info(f"✅ Current model: **{st.session_state.model_name}**")
    
    st.markdown("---")
    
    # Indexing Section - only show if model is loaded
    if st.session_state.model_loaded:
        st.subheader("4️⃣ Build/Load Index")
        
        # Check for existing indexes
        existing_indexes = find_saved_indexes(
            st.session_state.collection_name,
            st.session_state.model_name,
            base_dir=st.session_state.output_dir
        )
        
        if existing_indexes and not st.session_state.indexed:
            st.info(f"Found {len(existing_indexes)} existing index(es) for this collection and model")
            
            # Auto-load first index
            with st.spinner("Loading existing index..."):
                try:
                    coll_name = st.session_state.rag.collection_name
                    collection_folder_name = _sanitize_filename(coll_name)
                    first_index = existing_indexes[0]
                    index_base_path = os.path.join(st.session_state.output_dir, collection_folder_name, first_index)
                    
                    st.session_state.rag.set_index_paths(index_base_path)
                    num_chunks = st.session_state.rag.load_index()
                    st.session_state.indexed = True
                    st.success(f"✅ Loaded index with {num_chunks} chunks")
                    st.rerun()
                except ValueError as e:
                    if "Corrupted" in str(e):
                        st.warning(f"⚠️ {e}")
                        st.info("The index files appear to be corrupted. Please rebuild the index below.")
                    else:
                        st.warning(f"Could not auto-load index: {e}")
                except Exception as e:
                    st.warning(f"Could not auto-load index: {e}")
        
        # Show current status
        if st.session_state.indexed:
            st.success(f"✅ **Index loaded:** {len(st.session_state.rag.chunks)} chunks")
            st.info(f"📁 Path: {st.session_state.rag.index_path}")
            
            # Force reindex button
            if st.button("🔄 Force Reindex", type="secondary", use_container_width=True):
                with st.spinner("Rebuilding index..."):
                    pdf_progress_bar = st.progress(0)
                    encoding_progress_bar = st.progress(0)
                    
                    def progress_callback(stage, current, total, message):
                        if stage == 'pdf':
                            progress = current / total if total > 0 else 0
                            pdf_progress_bar.progress(progress, text=message)
                        elif stage == 'encoding':
                            progress = current / total if total > 0 else 0
                            encoding_progress_bar.progress(progress, text=message)
                    
                    try:
                        st.session_state.rag.set_index_paths()
                        num_chunks = st.session_state.rag.build_index(
                            force_rebuild=True,
                            progress_callback=progress_callback
                        )
                        
                        pdf_progress_bar.empty()
                        encoding_progress_bar.empty()
                        
                        st.success(f"✅ Index rebuilt with {num_chunks} chunks!")
                        st.rerun()
                    except Exception as e:
                        pdf_progress_bar.empty()
                        encoding_progress_bar.empty()
                        st.error(f"Error rebuilding index: {e}")
        else:
            # Build new index
            st.warning("⚠️ No index loaded. Please build an index to continue.")
            
            if st.button("🔨 Build Index", type="primary", use_container_width=True):
                pdf_progress_bar = st.progress(0)
                encoding_progress_bar = st.progress(0)
                
                def progress_callback(stage, current, total, message):
                    if stage == 'pdf':
                        progress = current / total if total > 0 else 0
                        pdf_progress_bar.progress(progress, text=message)
                    elif stage == 'encoding':
                        progress = current / total if total > 0 else 0
                        encoding_progress_bar.progress(progress, text=message)
                
                try:
                    st.session_state.rag.set_index_paths()
                    num_chunks = st.session_state.rag.build_index(
                        force_rebuild=False,
                        progress_callback=progress_callback
                    )
                    st.session_state.indexed = True
                    pdf_progress_bar.empty()
                    encoding_progress_bar.empty()
                    st.success(f"✅ Built index with {num_chunks} chunks!")
                    st.rerun()
                except Exception as e:
                    pdf_progress_bar.empty()
                    encoding_progress_bar.empty()
                    st.error(f"Error building index: {e}")
                    with st.expander("Show full error"):
                        st.exception(e)
    else:
        st.info("👆 Please load a model first")
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 Start Over", use_container_width=True):
        collections = st.session_state.get('collections', [])
        st.session_state.clear()
        st.session_state.collections_loaded = True
        st.session_state.collections = collections
        st.rerun()


def show_search_tab():
    """Search and highlight tab."""
    
    st.header("🔍 Search Your Library")
    
    # Show current configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        coll_name = st.session_state.rag.collection_name or "All Library"
        st.info(f"**Collection:** {coll_name}")
    with col2:
        st.info(f"**Model:** {st.session_state.model_name.split('/')[-1]}")
    with col3:
        st.info(f"**Chunks:** {len(st.session_state.rag.chunks)}")
    
    st.markdown("---")
    
    # Search interface
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your query",
            placeholder="What are the main findings about...",
            key="search_input"
        )
    with col2:
        threshold = st.number_input(
            "Threshold",
            min_value=0.1,
            max_value=2.0,
            value=1.2,
            step=0.1,
            help="Lower is stricter. Finds results within this L2 distance."
        )

    col_search, col_clear = st.columns([1, 4])
    with col_search:
        search_clicked = st.button("Search", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear Results", use_container_width=True):
            st.session_state.search_results = []
            st.session_state.current_index = 0
            st.rerun()

    if search_clicked and query:
        with st.spinner("Searching..."):
            st.session_state.search_results = st.session_state.rag.search(query, threshold=threshold)
            st.session_state.current_index = 0
            st.session_state.current_query = query
    
    # Display results
    if st.session_state.search_results:
        st.success(f"Found {len(st.session_state.search_results)} results for: *{st.session_state.current_query}*")
        
        # Navigation and actions
        col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 1])
        
        with col1:
            if st.button("⬅️ Previous"):
                # Wrap around: if at first result, go to last
                st.session_state.current_index = (st.session_state.current_index - 1) % len(st.session_state.search_results)
                st.rerun()
        
        with col2:
            st.markdown(f"**Result {st.session_state.current_index + 1} / {len(st.session_state.search_results)}**")
        
        with col3:
            if st.button("Next ➡️"):
                # Wrap around: if at last result, go to first
                st.session_state.current_index = (st.session_state.current_index + 1) % len(st.session_state.search_results)
                st.rerun()
        
        with col4:
            if st.button("📖 Open PDF"):
                chunk, _ = st.session_state.search_results[st.session_state.current_index]
                pdf_path = chunk.pdf_path
                try:
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', pdf_path])
                    elif sys.platform == 'win32':  # Windows
                        os.startfile(pdf_path)
                    else:  # Linux
                        subprocess.run(['xdg-open', pdf_path])
                    st.success(f"✅ Opened PDF at page {chunk.page_num + 1}")
                except Exception as e:
                    st.error(f"Could not open PDF: {e}")
        
        with col5:
            if st.button("💾 Highlight All"):
                # Group chunks by PDF
                pdfs_chunks = {}
                for chunk, _ in st.session_state.search_results:
                    if chunk.pdf_path not in pdfs_chunks:
                        pdfs_chunks[chunk.pdf_path] = []
                    pdfs_chunks[chunk.pdf_path].append(chunk)
                
                # Create output directory
                coll_name = st.session_state.rag.collection_name or "All_Library"
                output_dir = os.path.join(st.session_state.output_dir, coll_name, "highlighted")
                os.makedirs(output_dir, exist_ok=True)
                
                highlighted_paths = []
                progress_bar = st.progress(0)
                
                for idx, (pdf_path, chunks) in enumerate(pdfs_chunks.items()):
                    original_filename = os.path.basename(pdf_path)
                    name_without_ext = os.path.splitext(original_filename)[0]
                    output_filename = f"{name_without_ext}_highlighted.pdf"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    result_path = st.session_state.rag.highlight_pdf(chunks, output_path)
                    if result_path:
                        highlighted_paths.append(result_path)
                    
                    progress_bar.progress((idx + 1) / len(pdfs_chunks))
                
                progress_bar.empty()
                
                if highlighted_paths:
                    st.success(f"✅ Created {len(highlighted_paths)} highlighted PDF(s)")
                    with st.expander("Show files"):
                        for path in highlighted_paths:
                            st.text(f"📄 {os.path.basename(path)}")
                        st.text(f"📁 Location: {output_dir}")
                else:
                    st.error("Failed to create highlighted PDFs")
        
        # Current result display
        st.markdown("---")
        chunk, score = st.session_state.search_results[st.session_state.current_index]
        
        # Result card with color indicator
        color_hex = rgb_to_hex(chunk.color)
                
        st.subheader(f"📄 {chunk.title}")
        st.markdown(
            f"""**PDF**: {os.path.basename(chunk.pdf_path)}<br>
            **Page**: {chunk.page_num + 1} | 
            **Score**: {score:.4f} | 
            **Highlight Color**: <span style="color: {color_hex}; font-size: 20px;">●</span>""",
            unsafe_allow_html=True
        )
        
        # Content
        st.subheader("📝 Content")
        st.text_area(
            "Content of the selected chunk",
            value=chunk.text,
            height=300,
            disabled=True,
            label_visibility="collapsed",
            key=f"content_{st.session_state.current_index}"
        )
    
    elif query and search_clicked:
        st.info("No results found. Try adjusting the threshold or using different keywords.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Zotero RAG Navigator • Built with Streamlit</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()