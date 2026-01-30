import os
import sys
import yaml
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from zotero_rag import ZoteroRAG
from models import Answer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('run_from_config.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate YAML configuration file.
    
    Args:
        config_path: Path to YAML config file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set defaults
    config.setdefault('defaults', {})
    config['defaults'].setdefault('num_paraphrases', 2)
    config['defaults'].setdefault('retrieval_threshold', 0.7)
    config['defaults'].setdefault('rerank_threshold', 0.25)
    config['defaults'].setdefault('highlight_color', [1, 1, 0])
    config['defaults'].setdefault('question_type', 'general')
    config['defaults'].setdefault('custom_config', None)
    
    config.setdefault('output_base_dir', './output')
    config.setdefault('model_name', 'BAAI/bge-base-en-v1.5')
    config.setdefault('qa_model', 'deepset/roberta-base-squad2')
    config.setdefault('reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    config.setdefault('grobid_url', 'http://localhost:8070')
    config.setdefault('grobid_timeout', 180)
    config.setdefault('create_highlighted_pdfs', True)
    config.setdefault('rebuild_index', False)
    
    return config


def answer_to_dict(answer: Answer) -> Dict[str, Any]:
    """Convert Answer object to dictionary for JSON serialization.
    
    Args:
        answer: Answer object.
        
    Returns:
        Dictionary representation of the answer.
    """
    return {
        'text': answer.text,
        'context': answer.context,
        'pdf_path': answer.pdf_path,
        'page_num': answer.page_num,
        'item_key': answer.item_key,
        'title': answer.title,
        'section': answer.section,
        'start_char': answer.start_char,
        'end_char': answer.end_char,
        'score': answer.score,
        'query': answer.query,
        'color': answer.color,
        'retrieval_score': answer.retrieval_score,
        'rerank_score': answer.rerank_score
    }


def run_from_config(config_path: str) -> Dict[str, List[Answer]]:
    """Run ZoteroRAG pipeline from a YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Dictionary mapping questions to their answers.
    """
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Initialize ZoteroRAG
    logger.info("Initializing RAG system")
    
    # Determine source type
    source_type = config.get('source_type', 'zotero')
    
    if source_type == 'folder':
        rag = ZoteroRAG(
            source_type='folder',
            folder_path=config['folder_path'],
            model_name=config['model_name'],
            qa_model=config['qa_model'],
            reranker_model=config['reranker_model'],
            grobid_url=config['grobid_url'],
            grobid_timeout=config['grobid_timeout'],
            model_device=config.get('model_device'),
            encode_batch_size=config.get('encode_batch_size'),
            rerank_batch_size=config.get('rerank_batch_size'),
            output_base_dir=config['output_base_dir']
        )
    else:
        rag = ZoteroRAG(
            source_type='zotero',
            zotero_data_dir=config.get('zotero_data_dir'),
            collection_name=config.get('collection_name'),
            model_name=config['model_name'],
            qa_model=config['qa_model'],
            reranker_model=config['reranker_model'],
            grobid_url=config['grobid_url'],
            grobid_timeout=config['grobid_timeout'],
            model_device=config.get('model_device'),
            encode_batch_size=config.get('encode_batch_size'),
            rerank_batch_size=config.get('rerank_batch_size'),
            output_base_dir=config['output_base_dir']
        )
    
    # Load or build index
    rebuild_index = config.get('rebuild_index', False)
    
    if not rebuild_index and rag.index_exists():
        logger.info("Loading existing index")
        try:
            rag.load_index()
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Failed to load index: %s, building new index", e)
            logger.info("Building index from Zotero library")
            rag.build_index()
    else:
        logger.info("Building index from PDF source")
        rag.build_index()
    
    # Process questions
    questions = config.get('questions', [])
    if not questions:
        logger.warning("No questions specified in config file")
        return {}
    
    defaults = config['defaults']
    all_results = {}
    
    for i, question_config in enumerate(questions):
        question = question_config.get('question')
        if not question:
            logger.warning(f"Question {i+1} missing 'question' field, skipping")
            continue
        
        logger.info(f"Processing question {i+1}/{len(questions)}: {question}")
        
        # Get question-specific settings or use defaults
        num_paraphrases = question_config.get('num_paraphrases', defaults['num_paraphrases'])
        retrieval_threshold = question_config.get('retrieval_threshold', defaults['retrieval_threshold'])
        rerank_threshold = question_config.get('rerank_threshold', defaults['rerank_threshold'])
        highlight_color = question_config.get('highlight_color', defaults['highlight_color'])
        question_type = question_config.get('question_type', defaults.get('question_type', 'general'))
        
        # Convert highlight_color to tuple
        if highlight_color:
            highlight_color = tuple(highlight_color)
        
        # Get custom_config if provided
        custom_config = question_config.get('custom_config', defaults.get('custom_config'))
        
        # Get pre-defined paraphrases if provided
        paraphrases = question_config.get('paraphrases')
        question_variations = None
        if paraphrases:
            question_variations = [question] + paraphrases
            logger.info(f"Using {len(question_variations)} pre-defined question variations")
        
        # Answer the question
        answers = rag.answer_question(
            question=question,
            retrieval_threshold=retrieval_threshold,
            rerank_threshold=rerank_threshold,
            num_paraphrases=num_paraphrases,
            highlight_color=highlight_color,
            question_variations=question_variations,
            question_type=question_type,
            custom_config=custom_config
        )
        
        all_results[question] = answers
        logger.info(f"Found {len(answers)} answers for question: {question}")
        
        # Print top answers
        for j, answer in enumerate(answers[:3]):
            logger.info(f"  Answer {j+1}: {answer.text[:100]}... (score: {answer.score:.3f}, rerank: {answer.rerank_score:.3f})")
    
    # Create highlighted PDFs if requested
    if config.get('create_highlighted_pdfs', True):
        logger.info("Creating highlighted PDFs")
        
        # Group answers by PDF
        answers_by_pdf = {}
        for question, answers in all_results.items():
            for answer in answers:
                pdf_path = answer.pdf_path
                if pdf_path not in answers_by_pdf:
                    answers_by_pdf[pdf_path] = []
                answers_by_pdf[pdf_path].append(answer)
        
        # Create highlighted PDFs
        output_dir = os.path.join(config['output_base_dir'], 'highlighted_pdfs')
        os.makedirs(output_dir, exist_ok=True)
        
        for pdf_path, pdf_answers in answers_by_pdf.items():
            pdf_filename = Path(pdf_path).stem
            output_path = os.path.join(output_dir, f"{pdf_filename}_highlighted.pdf")
            
            logger.info(f"Highlighting {pdf_filename} with {len(pdf_answers)} answers")
            result_path = rag.highlight_pdf(pdf_answers, output_path)
            
            if result_path:
                logger.info(f"  Saved to: {result_path}")
            else:
                logger.warning(f"  Failed to create highlighted PDF")
    
    # Save results to JSON if requested
    output_file = config.get('output_results_file')
    if output_file:
        logger.info(f"Saving results to {output_file}")
        
        json_results = {}
        for question, answers in all_results.items():
            json_results[question] = [answer_to_dict(a) for a in answers]
        
        output_path = os.path.join(config['output_base_dir'], output_file)
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    return all_results


def main():
    """Main entry point for CLI usage."""
    if len(sys.argv) < 2:
        print("Usage: python -m zotero_rag.run_from_config <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        logger.info(f"Running ZoteroRAG from config: {config_path}")
        results = run_from_config(config_path)
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        total_answers = sum(len(answers) for answers in results.values())
        print(f"\nProcessed {len(results)} questions")
        print(f"Found {total_answers} total answers")
        
        for question, answers in results.items():
            print(f"\n{question}")
            print(f"  → {len(answers)} answers")
        
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Error running from config: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
