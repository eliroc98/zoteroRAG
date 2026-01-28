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
        st.session_state.output_dir = "/Users/lizzy/_research/transformers-cn/literature_output"
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
    if 'search_candidates' not in st.session_state:
        st.session_state.search_candidates = []
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
        st.session_state.model_name = "BAAI/bge-base-en-v1.5"
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

    col_device, col_encode_batch, col_rerank_batch = st.columns(3)
    
    with col_device:
        device_choice = st.selectbox(
            "Device",
            options=["auto", "cpu", "mps", "cuda"],
            index=["auto", "cpu", "mps", "cuda"].index(
                "auto" if st.session_state.model_device is None else st.session_state.model_device
            ),
            help="Select compute device. 'auto' picks MPS if available, else CPU."
        )
    
    with col_encode_batch:
        encode_batch_auto = st.checkbox(
            "Auto-detect encoding batch",
            value=st.session_state.get('encode_batch_auto', True),
            help="Auto-detect safe batch size (targets 75% memory usage)"
        )
        if not encode_batch_auto:
            encode_batch_size = st.number_input(
                "Encoding batch size",
                min_value=1, max_value=256, value=st.session_state.get('encode_batch_size', 8),
                help="Manual batch size for encoding"
            )
        else:
            encode_batch_size = None
            st.session_state.encode_batch_auto = True
    
    with col_rerank_batch:
        rerank_batch_auto = st.checkbox(
            "Auto-detect rerank batch",
            value=st.session_state.get('rerank_batch_auto', True),
            help="Auto-detect safe batch size (targets 75% memory usage)"
        )
        if not rerank_batch_auto:
            rerank_batch_size = st.number_input(
                "Reranking batch size",
                min_value=1, max_value=256, value=st.session_state.get('rerank_batch_size', 8),
                help="Manual batch size for reranking"
            )
        else:
            rerank_batch_size = None
            st.session_state.rerank_batch_auto = True
    
    if st.button("📥 Load Model", type="primary", use_container_width=True):
        if model_input:
            with st.spinner(f"Loading model: {model_input}..."):
                try:
                    st.session_state.model_name = model_input
                    st.session_state.grobid_url = grobid_url
                    st.session_state.model_device = None if device_choice == "auto" else device_choice
                    st.session_state.encode_batch_size = encode_batch_size
                    st.session_state.rerank_batch_size = rerank_batch_size
                    st.session_state.rag = ZoteroRAG(
                        collection_name=st.session_state.collection_name,
                        model_name=model_input,
                        grobid_url=grobid_url,
                        output_base_dir=st.session_state.output_dir,
                        model_device=st.session_state.model_device,
                        encode_batch_size=encode_batch_size,
                        rerank_batch_size=rerank_batch_size
                    )
                    st.session_state.model_loaded = True
                    st.session_state.indexed = False  # Reset indexed status
                    
                    # Show configuration summary
                    batch_info = []
                    if encode_batch_size is None:
                        batch_info.append("Encoding: Auto-detect")
                    else:
                        batch_info.append(f"Encoding: {encode_batch_size}")
                    if rerank_batch_size is None:
                        batch_info.append("Reranking: Auto-detect")
                    else:
                        batch_info.append(f"Reranking: {rerank_batch_size}")
                    
                    st.success(f"✅ Model loaded: {model_input}\n\n**Batch sizes:** {' | '.join(batch_info)}")
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
            st.success(f"✅ **Index loaded:** {len(st.session_state.rag.paragraphs)} paragraphs")
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
        st.info(f"**Paragraphs:** {len(st.session_state.rag.paragraphs)}")
    
    st.markdown("---")
    
    # Search interface - split into query and 3 parameter columns
    col_query, col_retrieval, col_rerank, col_qa = st.columns([3, 1, 1, 1])
    
    with col_query:
        query = st.text_input(
            "Enter your question",
            placeholder="What are the main findings about...",
            key="search_input"
        )
        
    with col_retrieval:
        retrieval_threshold = st.number_input(
            "1. Retrieval Dist.",
            min_value=0.1, max_value=10.0, value=2.0, step=0.1,
            help="Stage 1 (FAISS): Higher = more paragraphs retrieved."
        )

    with col_rerank:
        rerank_threshold = st.number_input(
            "2. Rerank Thresh.",
            min_value=0.0, max_value=1.0, value=0.25, step=0.05,
            help="Stage 2 (CrossEncoder): Minimum semantic similarity score (0.0-1.0) to keep a paragraph."
        )

    with col_qa:
        qa_score_threshold = st.number_input(
            "3. QA Conf.",
            min_value=0.0, max_value=1.0, value=0.0, step=0.1,
            help="Stage 3 (QA Model): Minimum answer confidence."
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
        import time
        
        # Create separate progress tracking for each stage
        st.markdown("#### Processing Pipeline")
        
        rerank_status = st.empty()
        rerank_progress_bar = st.progress(0)
        
        qa_status = st.empty()
        qa_progress_bar = st.progress(0)
        
        # Track timing for each stage
        rerank_start_time = [None]  # Use list for mutability in nested function
        qa_start_time = [None]
        
        def rerank_callback(current, total, message):
            if rerank_start_time[0] is None:
                rerank_start_time[0] = time.time()
            
            if total > 0:
                progress = current / total
                rerank_progress_bar.progress(progress)
                
                # Calculate time estimates
                elapsed = time.time() - rerank_start_time[0]
                if current > 0 and progress < 1.0:
                    estimated_total = elapsed / progress
                    remaining = estimated_total - elapsed
                    time_info = f"⏱️ Elapsed: {elapsed:.1f}s | Remaining: ~{remaining:.1f}s"
                elif progress >= 1.0:
                    time_info = f"⏱️ Completed in {elapsed:.1f}s"
                else:
                    time_info = ""
            else:
                rerank_progress_bar.progress(0)
                time_info = ""
            
            rerank_status.text(f"🔄 Stage 1 - Reranking: {message} {time_info}")
        
        def qa_callback(current, total, message):
            if qa_start_time[0] is None:
                qa_start_time[0] = time.time()
            
            if total > 0:
                progress = current / total
                qa_progress_bar.progress(progress)
                
                # Calculate time estimates
                elapsed = time.time() - qa_start_time[0]
                if current > 0 and progress < 1.0:
                    estimated_total = elapsed / progress
                    remaining = estimated_total - elapsed
                    time_info = f"⏱️ Elapsed: {elapsed:.1f}s | Remaining: ~{remaining:.1f}s"
                elif progress >= 1.0:
                    time_info = f"⏱️ Completed in {elapsed:.1f}s"
                else:
                    time_info = ""
            else:
                qa_progress_bar.progress(0)
                time_info = ""
            
            qa_status.text(f"🤖 Stage 2 - QA Extraction: {message} {time_info}")
        
        try:
            st.session_state.search_results = st.session_state.rag.answer_question(
                question=query,
                retrieval_threshold=retrieval_threshold,
                qa_score_threshold=qa_score_threshold,
                rerank_threshold=rerank_threshold,
                progress_callback=qa_callback,
                rerank_callback=rerank_callback
            )
            
            # Mark completion
            rerank_progress_bar.progress(1.0)
            qa_progress_bar.progress(1.0)
            rerank_status.text("🔄 Stage 1 - Reranking: ✓ Complete!")
            qa_status.text("🤖 Stage 2 - QA Extraction: ✓ Complete!")
            
            st.session_state.search_candidates = getattr(st.session_state.rag, "last_candidates", [])
            st.session_state.current_index = 0
            st.session_state.current_query = query
        except Exception as e:
            st.error(f"❌ Search failed: {e}")
            st.session_state.search_results = []
            with st.expander("Show full error"):
                st.exception(e)
        finally:
            time.sleep(1.0)
            rerank_status.empty()
            rerank_progress_bar.empty()
            qa_status.empty()
            qa_progress_bar.empty()
    
    # Display results
    if st.session_state.search_results:
        st.success(f"✅ Found {len(st.session_state.search_results)} answers for: *{st.session_state.current_query}*")
        
        # Navigation and actions
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 1])
        
        with col1:
            if st.button("⬅️ Previous"):
                # Wrap around: if at first result, go to last
                st.session_state.current_index = (st.session_state.current_index - 1) % len(st.session_state.search_results)
                st.rerun()
        
        with col2:
            st.markdown(f"**Answer {st.session_state.current_index + 1} / {len(st.session_state.search_results)}**")
        
        with col3:
            if st.button("Next ➡️"):
                # Wrap around: if at last result, go to first
                st.session_state.current_index = (st.session_state.current_index + 1) % len(st.session_state.search_results)
                st.rerun()
        
        with col4:
            if st.button("📖 Open PDF"):
                answer = st.session_state.search_results[st.session_state.current_index]
                pdf_path = answer.pdf_path
                try:
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', pdf_path])
                    elif sys.platform == 'win32':  # Windows
                        os.startfile(pdf_path)
                    else:  # Linux
                        subprocess.run(['xdg-open', pdf_path])
                    st.success(f"✅ Opened PDF at page {answer.page_num + 1}")
                except Exception as e:
                    st.error(f"Could not open PDF: {e}")
        
        with col5:
            if st.button("💾 Highlight All"):
                # Answers already have the highlighting info
                coll_name = st.session_state.rag.collection_name or "All_Library"
                output_dir = os.path.join(st.session_state.output_dir, coll_name, "highlighted")
                os.makedirs(output_dir, exist_ok=True)
                
                # Group answers by PDF
                pdfs_answers = {}
                for answer in st.session_state.search_results:
                    if answer.pdf_path not in pdfs_answers:
                        pdfs_answers[answer.pdf_path] = []
                    pdfs_answers[answer.pdf_path].append(answer)
                
                highlighted_paths = []
                progress_bar = st.progress(0)
                
                for idx, (pdf_path, answers) in enumerate(pdfs_answers.items()):
                    original_filename = os.path.basename(pdf_path)
                    name_without_ext = os.path.splitext(original_filename)[0]
                    output_filename = f"{name_without_ext}_highlighted.pdf"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    result_path = st.session_state.rag.highlight_pdf(answers, output_path)
                    if result_path:
                        highlighted_paths.append(result_path)
                    
                    progress_bar.progress((idx + 1) / len(pdfs_answers))
                
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
        answer = st.session_state.search_results[st.session_state.current_index]
        
        # Result card with color indicator
        color_hex = rgb_to_hex(answer.color)
                
        st.subheader(f"📄 {answer.title}")
        st.markdown(
            f"""**PDF**: {os.path.basename(answer.pdf_path)}<br>
            **Page**: {answer.page_num + 1} | 
            **Section**: {answer.section or 'Unknown'} | 
            **Retrieval Score**: {answer.retrieval_score:.4f} | 
            **QA Score**: {answer.score:.4f} | 
            **Highlight Color**: <span style="color: {color_hex}; font-size: 20px;">●</span>""",
            unsafe_allow_html=True
        )
        
        # Answer display
        st.subheader("💡 Answer")
        st.info(answer.text)
        
        # Context
        st.subheader("📝 Context (Full Paragraph)")
        st.text_area(
            "Full paragraph containing the answer",
            value=answer.context,
            height=250,
            disabled=True,
            label_visibility="collapsed",
            key=f"context_{st.session_state.current_index}"
        )
    
    elif query and search_clicked:
        st.warning("⚠️ No results found")
        st.info("""
        **Possible reasons:**
        - QA score threshold too high (try 0.0)
        - Retrieval threshold too low (try increasing to 3.0-5.0)
        - No semantically similar paragraphs found
        - Question format doesn't match extractive QA style
        
        **Try:**
        - Lower the QA Score Threshold to 0.0
        - Increase Retrieval Threshold to 3.0 or higher
        - Rephrase as a specific question (e.g., "What is X?" instead of "Tell me about X")
        """)
    elif not query:
        st.info("👆 Enter a question above and click Search to get started")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Zotero RAG Navigator • Built with Streamlit</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()