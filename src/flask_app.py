'''
## Proof of Concept for Climate Policy Analysis App (Python + Flask + Unstructured + PyMuPDF + Semantic Search)

This Flask application serves as a proof of concept for analyzing climate policy documents.
It uses various tools and libraries to process PDFs, extract content, and perform semantic search.
Key Features:
 - Upload and process PDF documents
 - Extract text and metadata from PDFs
 - Perform semantic search using vector embeddings
 - Interact with Azure OpenAI models for advanced language processing

'''

from fileinput import filename
from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify
import fitz  # PyMuPDF
import os
import tempfile
import json
import pandas as pd
import uuid
import queue
import threading
import logging
import time
import random
import requests
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlparse
from werkzeug.utils import secure_filename
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import HumanMessage
from langchain_community.vectorstores import FAISS # https://python.langchain.com/docs/integrations/vectorstores/faiss/
from langchain.docstore.document import Document
from langchain_community.callbacks import get_openai_callback
from openai import InternalServerError, RateLimitError
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from zipfile import ZipFile
import io
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
import traceback
import sys
from unstructured.cleaners.core import clean_extra_whitespace, clean_dashes, clean_bullets
import re
import unicodedata
import gc
import traceback
import psutil

try:
    from .quotation_utils import get_quotes
    from .classification_utils import tagging_classifier_quotes
    from .general_utils import create_highlighted_pdf, _sanitize_for_json
except ImportError:
    # Fallback for direct execution
    from quotation_utils import get_quotes
    from classification_utils import tagging_classifier_quotes
    from general_utils import create_highlighted_pdf, _sanitize_for_json

LOCAL = False  # Set to True for local development

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir() # Temporary folder for file uploads

# Set up logging - GLOBAL app log (for startup, errors, etc.)
global_log_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'app_logs')
os.makedirs(global_log_folder, exist_ok=True)
global_log_file = os.path.join(global_log_folder, 'app.log')

logging.basicConfig(
    filename=global_log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info("============ App started ============")
logging.info(f"Global log file: {global_log_file}")

# Store for tracking processing tasks
processing_tasks = {}

# Environment variable setup for LangChain tracing
# Uncomment the following lines to enable tracing with LangChain (requires valid API key)
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
#os.environ["LANGCHAIN_PROJECT"] = "climate_policy_pipeline"
os.environ["LANGCHAIN_TRACING_V2"] = "false" # deactivate tracing for now, as api key is required to be updated to v2: "error":"Unauthorized: Using outdated v1 api key. Please use v2 api key."

# Load environment variables from .env file - for local development
if LOCAL:
    env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    logging.info(f"Loaded environment variables from {env_path}")

# Azure OpenAI Setup 
LLM = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    model_version='2024-07-18',
    temperature=0
)

# Embedding function
# These embeddings are used for semantic search and similarity comparisons.
EMBEDDING = AzureOpenAIEmbeddings(
    openai_api_key= os.environ["AZURE_OPENAI_API_KEY"],
    deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
    )

# Default semantic search query (token-based)
DEFAULT_QUERY = ["transport  vehicles  emissions  BA  electric U  GH  reduction  public G  compared  scenario  transportation  reduce  CO  levels  net es  zero  fuel  vehicle  em  passenger  road  mobility  veh  target  new idad CO  Mt  car ision  transporte  rail  cars  fleet  buses  fuels  traffic  efficiency ículos ar ct e  gas  greenhouse  redu  freight  d  l  share  km o  bio  achieve os  elé els  hydrogen  urban  infrastructure  electr The  hybrid  relative  charging  neutrality eq  é ici )  least  total ado  emission  vé  standards én  aims  e  ambition ’  modes il  carbon  shift as  neutral fu  bus  EV  ré  mov  condition hic  sales  million cción  inter  año  modal  maritime  system  diesel  público  kt  network ules  alternative  cities  percent  heavy re  conditional  Transport  improvement -em  Electric RT  level  use nel  transit  roads  in  light ibles  energ  year rica  goal  aviation  per missions  long  powered  European  consumption arbon ric  lanes  vo  part  walking  sharing  rapport ación  t  bicycle  motor  stations  infra  s duction ov a To  sc  railways  cent  private ías  reductions ). ual r  achieved ada -m condition  élect  ef  resilience  proof  infrastructure"]

# Define a sample taxonomy
TAXONOMY = ["Transport",
            "Energy",
            "Net Zero",
            "Mitigation",
            "Adaptation",
            "Conditional",
            "Target",
            "Measure"
            ]

def get_task_results_folder(task_id):
    """
    Get the results folder path for a specific task.
    
    Args:
        task_id (str): Unique task identifier
        
    Returns:
        str: Path to task-specific results folder
    """
    return os.path.join(app.config['UPLOAD_FOLDER'], f'results_{task_id}')

def cleanup_old_task_results(age_hours=24):
    """
    Clean up task results older than specified hours.
    
    Args:
        age_hours (int): Age in hours after which results are deleted
    """
    try:
        results_base = app.config['UPLOAD_FOLDER']
        current_time = time.time()
        age_seconds = age_hours * 3600
        cleaned_count = 0
        
        for folder_name in os.listdir(results_base):
            if folder_name.startswith('results_'):
                folder_path = os.path.join(results_base, folder_name)
                if os.path.isdir(folder_path):
                    folder_age = current_time - os.path.getctime(folder_path)
                    if folder_age > age_seconds:
                        logging.info(f"Cleaning up old results: {folder_name} (age: {folder_age/3600:.1f} hours)")
                        import shutil
                        shutil.rmtree(folder_path)
                        cleaned_count += 1
        
        if cleaned_count > 0:
            logging.info(f"Cleanup completed: {cleaned_count} old result folders removed")
                        
    except Exception as e:
        logging.error(f"Cleanup error: {e}")

def download_pdf_from_url(url, save_dir):
    """
    Downloads a PDF file from a URL and saves it to the specified directory.
    
    Args:
        url (str): The URL of the PDF file
        save_dir (str): Directory to save the downloaded file
        
    Returns:
        tuple: (file_path, filename) if successful, (None, None) if failed
    """
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.netloc:
            raise ValueError("Invalid URL provided")
        
        # Set up headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request with a timeout
        logging.info(f"Downloading PDF from URL: {url}")
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # Check if the response is actually a PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' not in content_type:
            # Try to determine from URL extension as fallback
            if not url.lower().endswith('.pdf'):
                raise ValueError("URL does not point to a PDF file")
        
        # Generate filename from URL
        parsed_path = urlparse(url).path
        if parsed_path:
            filename = os.path.basename(parsed_path)
            if not filename.endswith('.pdf'):
                filename += '.pdf'
        else:
            filename = 'downloaded_document.pdf'
        
        # Ensure filename is secure
        filename = secure_filename(filename)
        if not filename:
            filename = 'downloaded_document.pdf'

        # Limit filename length to 90 characters (including extension)
        MAX_FILENAME_LENGTH = 90
        name, ext = os.path.splitext(filename)
        ext = ext if ext else '.pdf'
        # Remove unnecessary characters (spaces, underscores, dashes, etc.) if too long
        if len(filename) > MAX_FILENAME_LENGTH:
            # Remove common URL encodings and unwanted characters
            name = re.sub(r'%20|%27|%28|%29|[\s_-]+', '', name)
            # Truncate if still too long
            name = name[:MAX_FILENAME_LENGTH - len(ext)]
            filename = name + ext
        
        # Save the file
        file_path = os.path.join(save_dir, filename)
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify the file was downloaded and has content
        if os.path.getsize(file_path) == 0:
            os.remove(file_path)
            raise ValueError("Downloaded file is empty")
        
        logging.info(f"Successfully downloaded PDF: {filename} ({os.path.getsize(file_path)} bytes)")
        return file_path, filename
        
    except requests.exceptions.Timeout:
        logging.error(f"Timeout downloading PDF from URL: {url}")
        raise ValueError("Download timeout - the server took too long to respond")
    except requests.exceptions.ConnectionError:
        logging.error(f"Connection error downloading PDF from URL: {url}")
        raise ValueError("Connection error - could not reach the server")
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error downloading PDF from URL: {url} - {e}")
        raise ValueError(f"HTTP error {e.response.status_code} - could not download the file")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error downloading PDF from URL: {url} - {e}")
        raise ValueError(f"Request error - {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error downloading PDF from URL: {url} - {e}")
        raise ValueError(f"Unexpected error - {str(e)}")

def clean_elements(elements, apply_cleaning=True):
    """
    Clean and normalize text elements extracted from PDF documents.
    This function applies general text cleaning operations that work across different document types.
    
    Args:
        elements (list): List of unstructured elements from partition_pdf
        apply_cleaning (bool): Whether to apply cleaning functions. Default: True.
        
    Returns:
        list: A list of dictionaries containing cleaned text and metadata
    """
    if not apply_cleaning:
        # Return elements without cleaning
        content = []
        for el in elements:
            if el.text and el.text.strip():
                content.append({
                    "text": el.text.strip(),
                    "metadata": el.metadata.to_dict() if el.metadata else {},
                    "element_type": getattr(el, 'category', 'Unknown'),
                    "cleaned": False
                })
        return content
    
    content = []
    cleaning_stats = {
        "total_elements": len(elements),
        "processed_elements": 0,
        "characters_before": 0,
        "characters_after": 0,
        "empty_after_cleaning": 0
    }
    
    for el in elements:
        if el.text and el.text.strip():
            original_text = el.text.strip()
            cleaning_stats["characters_before"] += len(original_text)
            
            try:
                # Apply general text cleaning
                cleaned_text = apply_general_cleaning(original_text)
                cleaning_stats["characters_after"] += len(cleaned_text)
                
                # Only keep non-empty cleaned text
                if cleaned_text and len(cleaned_text.strip()) >= 10:  # Minimum length threshold
                    content.append({
                        "text": cleaned_text,
                        "original_text": original_text,
                        "metadata": el.metadata.to_dict() if el.metadata else {},
                        "element_type": getattr(el, 'category', 'Unknown'),
                        "cleaned": True,
                        "char_reduction": len(original_text) - len(cleaned_text)
                    })
                    cleaning_stats["processed_elements"] += 1
                else:
                    cleaning_stats["empty_after_cleaning"] += 1
                    
            except Exception as e:
                # If cleaning fails, keep original text
                logging.warning(f"Cleaning failed for element, keeping original: {e}")
                content.append({
                    "text": original_text,
                    "metadata": el.metadata.to_dict() if el.metadata else {},
                    "element_type": getattr(el, 'category', 'Unknown'),
                    "cleaned": False
                })
                cleaning_stats["processed_elements"] += 1
    
    # Log cleaning statistics
    char_reduction = cleaning_stats["characters_before"] - cleaning_stats["characters_after"]
    reduction_percent = (char_reduction / cleaning_stats["characters_before"] * 100) if cleaning_stats["characters_before"] > 0 else 0
    
    logging.info(f"Text cleaning completed:")
    logging.info(f"  - Processed: {cleaning_stats['processed_elements']}/{cleaning_stats['total_elements']} elements")
    logging.info(f"  - Character reduction: {char_reduction} chars ({reduction_percent:.1f}%)")
    logging.info(f"  - Empty after cleaning: {cleaning_stats['empty_after_cleaning']} elements")
    
    return content

def apply_general_cleaning(text):
    """
    Apply general text cleaning operations suitable for any PDF document type.
    
    Args:
        text (str): Raw text extracted from PDF
        
    Returns:
        str: Cleaned and normalized text
    """
    if not text or not isinstance(text, str):
        return text
    
    # Step 1: Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Step 2: Replace tab characters with single spaces
    text = re.sub(r'\t+', ' ', text)
    
    # Step 3: Fix hyphenated words split across lines
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # Step 4: Clean up common PDF artifacts
    text = re.sub(r'\f', ' ', text)  # Form feed characters
    text = re.sub(r'\x0c', ' ', text)  # Page break characters
    text = re.sub(r'\u00a0', ' ', text)  # Non-breaking spaces
    
    # Step 5: Remove standalone page numbers and common headers/footers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone numbers
    text = re.sub(r'(?i)^\s*page\s+\d+(\s+of\s+\d+)?\s*$', '', text, flags=re.MULTILINE)
    
    # Step 6: Clean up bullet points and list markers
    text = clean_bullets(text)
    
    # Step 7: Normalize dashes
    text = clean_dashes(text)
    
    # Step 8: Fix spacing around punctuation
    text = re.sub(r'\s*([,.;:])\s*', r'\1 ', text)
    text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
    text = re.sub(r'\s*([()[\]{}])\s*', r' \1 ', text)
    
    # Step 9: Clean up number formatting
    text = re.sub(r'(\d+)\s*[.,]\s*(\d+)', r'\1.\2', text)  # Decimal numbers
    text = re.sub(r'(\d+)\s*%', r'\1%', text)  # Percentages
    text = re.sub(r'(\d+)\s*°', r'\1°', text)  # Degrees
    
    # Step 10: Reconnect sentences split inappropriately
    # Join lines that end with lowercase and start with lowercase
    text = re.sub(r'([a-z])\s*\n\s*([a-z])', r'\1 \2', text)
    
    # Step 11: Remove excessive whitespace
    text = clean_extra_whitespace(text)
    
    # Step 12: Clean up multiple newlines but preserve paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)  # Trailing spaces before newlines
    text = re.sub(r'\n[ \t]+', '\n', text)  # Leading spaces after newlines
    
    # Step 13: Remove very short standalone words/characters on separate lines
    text = re.sub(r'\n\s*[a-zA-Z]\s*\n', '\n', text)  # Single characters
    text = re.sub(r'\n\s*[ivxlcdm]+\s*\n', '\n', text, flags=re.IGNORECASE)  # Roman numerals
    
    # Step 14: Clean up common abbreviations and acronyms spacing
    # Fix spaced-out acronyms (e.g., "U S A" -> "USA")
    text = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z])\b', r'\1\2\3', text)
    text = re.sub(r'\b([A-Z])\s+([A-Z])\b', r'\1\2', text)
    
    # Step 15: Fix common chemical/scientific notation
    text = re.sub(r'CO\s*2(?!\w)', 'CO2', text)  # Carbon dioxide
    text = re.sub(r'H\s*2\s*O(?!\w)', 'H2O', text)  # Water
    text = re.sub(r'CH\s*4(?!\w)', 'CH4', text)  # Methane
    text = re.sub(r'N\s*2\s*O(?!\w)', 'N2O', text)  # Nitrous oxide
    
    # Step 16: Clean up URLs and email addresses that might be broken
    text = re.sub(r'(https?://)\s+', r'\1', text)
    text = re.sub(r'(\w+@)\s+(\w+)', r'\1\2', text)
    
    # Step 17: Final cleanup
    text = text.strip()
    
    return text

# Update the existing extract_text_with_unstructured function (around line 234)
def extract_text_with_unstructured(pdf_path, task_id, min_lenght=10, max_length=10000, apply_cleaning=True):
    """
    Extracts text and metadata from a PDF file using the `partition_pdf` function.
    Now includes general text cleaning using clean_elements() and task-specific storage.

    Args:
        pdf_path (str): The file path to the PDF document to be processed.
        task_id (str): Unique task identifier for folder structure
        min_lenght (int): The minimum length of elements, up to which they are always chunked, to reduce processing time.
        max_length (int): The maximum length of chunks to be extracted from each element. Larger chunks are split into smaller ones.
        apply_cleaning (bool): Whether to apply text cleaning functions. Default: True.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - "text" (str): The extracted text content from a PDF element.
            - "metadata" (dict): Metadata associated with the PDF element, if available.
            - "original_text" (str): The original unclean text (if cleaning was applied).
    """
    try:
        logging.info(f"Starting text extraction from: {pdf_path}")
        
        # Extract elements using unstructured
        elements = partition_pdf(
            filename=pdf_path,
            #strategy='hi_res', # requires Poppler installation
            coordinates=True,  # return bounding box coordinates for each element extracted via OCR
            #infer_table_structure=True,  # infer table structure
            #extract_images_in_pdf=True,  # extract images in PDF
            languages=['eng', 'ara', 'chi', 'fre', 'rus', 'spa'],  # languages used by the United Nations
            #extract_image_block_types=["Image", "Table"],  # The types of elements to extract, for use in extracting image blocks as base64 encoded data stored in metadata fields
        )
        
        logging.info(f"Extracted {len(elements)} elements from PDF")
        
        # Chunk the elements
        elements = chunk_by_title(
            elements=elements,
            multipage_sections=True,  # Default
            combine_text_under_n_chars=min_lenght,  # Specifying 0 for this argument suppresses combining of small chunks
            new_after_n_chars=max_length,  # Specifying 0 for this argument causes each element to appear in a chunk by itself (although an element with text longer than max_characters will be still be split into two or more chunks)
            max_characters=max_length  # Cut off new sections after reaching a length of n chars (hard max). Default: 500
        )
        
        logging.info(f"Chunked into {len(elements)} elements")

        # Apply general cleaning using clean_elements
        content = clean_elements(elements, apply_cleaning=apply_cleaning)
        
        # Save content as JSON in the by-products folder
        by_products_folder = os.path.join(get_task_results_folder(task_id), 'by-products')
        os.makedirs(by_products_folder, exist_ok=True)
        
        # Fix the duplicate path issue and add cleaning suffix
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        json_suffix = "_partition_cleaned.json" if apply_cleaning else "_partition_raw.json"
        json_path = os.path.join(by_products_folder, f"{base_filename}{json_suffix}")
        
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(content, json_file, ensure_ascii=False, indent=4)
        
        logging.info(f"Saved extracted content to: {json_path}")
        
        return content
        
    except Exception as e:
        logging.error(f"Error in extract_text_with_unstructured: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise

def semantic_search_FAISS(text_chunks, queries, filename, task_id, embedding_model, sim_threshold=0.69, k=20, max_retries=3): 
    """
    Perform a semantic search over a collection of text chunks using a vector store with FAISS.
    The vector store is saved for future use.

    Args:
        text_chunks (list of dict): A list of dictionaries where each dictionary represents a text chunk.
            Each dictionary must have a 'text' key containing the text content, and optionally a 'metadata' key
            containing additional metadata (e.g., page number).
        queries (list of str): A list of query strings to search for in the text chunks.
        filename (str): The name of the file being processed (used for naming the vector store)
        task_id (str): Unique task identifier
        embedding_model: The embedding model to use
        sim_threshold (float, optional): The minimum cosine similarity score to consider a match. Defaults to 0.69.
        k (int, optional): The number of top matches to retrieve for each query. Defaults to 20.
        max_retries (int, optional): The maximum number of retries for creating embeddings in case of service unavailability. Defaults to 3.

    Returns:
        list of dict: A list of dictionaries representing the matched results. Each dictionary contains:
            - "text" (str): The content of the matched text chunk.
            - "page" (int or str): The page number from the metadata, or "N/A" if not provided.
            - "score" (float): The cosine similarity score of the match (higher is better, threshold of sim_threshold is applied).
    """
    documents = [Document(page_content=chunk['text'], metadata=chunk.get("metadata", {})) for chunk in text_chunks]
    
    # Create the vector store directory if it doesn't exist
    vector_store_dir = os.path.join(get_task_results_folder(task_id), 'vector_stores')
    os.makedirs(vector_store_dir, exist_ok=True)
    
    # Create a unique name for this document's vector store
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    vector_store_path = os.path.join(vector_store_dir, f"{base_filename}_vectorstore")
    
    # Create and save the vector store with retry logic
    for attempt in range(max_retries):
        try:
            vectorstore = FAISS.from_documents(documents, embedding_model)
            vectorstore.save_local(vector_store_path)
            break  # Success, exit retry loop
        except InternalServerError as e:
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                logging.warning(f"Azure OpenAI service temporarily unavailable (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to create embeddings after {max_retries} attempts: {e}")
                raise  # Re-raise the exception after all retries failed

    # Perform similarity search for each query with minimum score threshold
    matches = []
    for query in queries:
        # Getting more results initially
        results = vectorstore.similarity_search_with_relevance_scores(query, k=len(text_chunks))
        # Getting results with scores, filter highest matches 
        for res, score in results:
            if score >= sim_threshold:  # FAISS returns similarity score, higher is better
                matches.append({
                    "text": res.page_content,
                    "filename": res.metadata.get("filename", "N/A"),
                    "page": res.metadata.get("page_number", "N/A"),
                    "score": score
                })
    return matches

def semantic_search_Chroma(text_chunks, queries, filename, task_id, embedding_model, sim_threshold=0.65, k=20):
    """
    Perform semantic search using Chroma with task-specific vector store.

    Args:
        text_chunks (list of dict): Text chunks with metadata
        queries (list of str): Query strings
        filename (str): Original filename
        task_id (str): Unique task identifier
        embedding_model: The embedding model to use
        sim_threshold (float): Similarity threshold
        k (int): Number of results to retrieve

    Returns:
        list of dict: Matched results with text, page, and score
    """
    
    documents = []
    for chunk in text_chunks:
        metadata = chunk.get("metadata", {})
        if isinstance(metadata, dict):
            try:
                metadata = filter_complex_metadata(metadata)
            except Exception as e:
                metadata = {k: str(v) for k, v in metadata.items() if k != "metadata"}
        else:
            metadata = {"raw_metadata": str(metadata)}
        
        documents.append(Document(page_content=chunk['text'], metadata=metadata))
    
    # Create task-specific vector store directory
    vector_store_dir = os.path.join(get_task_results_folder(task_id), 'vector_stores')
    os.makedirs(vector_store_dir, exist_ok=True)
    
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    vector_store_path = os.path.join(vector_store_dir, f"{base_filename}_chroma")
    
    # Create and save the vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=base_filename,
        persist_directory=vector_store_path
    )
    vectorstore.persist()

    # Perform similarity search
    matches = []
    for query in queries:
        results = vectorstore.similarity_search_with_score(query, k=len(text_chunks))
        
        for res, score in results:
            if score >= sim_threshold:
                matches.append({
                    "text": res.page_content,
                    "filename": res.metadata.get("filename", "N/A"),
                    "page": res.metadata.get("page_number", "N/A"),
                    "score": score
                })
        
    return matches

def extract_quotes(matches, filename, task_id):
    """
    Process semantic search matches to extract quotes using get_quotes function.
    
    Args:
        matches (list): List of matches from semantic search
        filename (str): Base name of the file being processed
        task_id (str): Unique task identifier
    
    Returns:
        list: Original matches with extracted quotes added
        int: Total tokens used during the extraction process
    """
    try:
        # Convert matches into the dictionary format needed for get_quotes
        doc_dict = {}
        for i, match in enumerate(matches):
            doc_dict[str(i)] = {
                'filename': match["filename"],
                'content': match["text"],
                'type': 'text',  # Assuming all are text entries
                'page_number': match["page"],
                'score': match["score"]
            }
        
        logging.info(f"Prepared {len(doc_dict)} matches for quote extraction")
        
        # Estimate API calls needed
        estimated_calls = len(doc_dict) * 2  # extraction + revision per chunk
        logging.info(f"Estimated Azure OpenAI API calls needed: {estimated_calls}")
        
        if estimated_calls > 50:
            logging.warning(f"High API call count ({estimated_calls}) may trigger rate limits")
        
        # Create the by-products folder if it doesn't exist
        by_products_folder = os.path.join(get_task_results_folder(task_id), 'by-products')
        os.makedirs(by_products_folder, exist_ok=True)
        
        # Extract quotes using the provided utility with enhanced error handling
        start_time = time.time()
        with get_openai_callback() as cb:
            quotes_dict = get_quotes(LLM, doc_dict)
            total_tokens = cb.total_tokens
        
        extraction_time = time.time() - start_time
        logging.info(f"Quote extraction completed in {extraction_time:.2f} seconds")
        logging.info(f"API calls per second: {estimated_calls/extraction_time:.2f}")
        logging.info(f"Total tokens used for quote extraction: {total_tokens}")
            
        # Save the quotes to a JSON file
        quotes_path = os.path.join(by_products_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_extracted-quotes.json")
        with open(quotes_path, 'w', encoding='utf-8') as f:
            json.dump(quotes_dict, f, ensure_ascii=False, indent=4)
        
        return quotes_dict, total_tokens
        
    except Exception as e:
        logging.error(f"Error in extract_quotes function: {e}")
        logging.error(f"extract_quotes traceback: {traceback.format_exc()}")
        
        # Check for rate limiting specifically
        if "rate limit" in str(e).lower():
            raise Exception(f"Azure OpenAI rate limit: {e}")
        else:
            raise

def highlight_terms_in_pdf(pdf_path, query_terms):
    """
    Highlights specified terms in a PDF document using PyMuPDF.
    
    TODO: This function should highlight only found quotes.
    """
    doc = fitz.open(pdf_path)
    for page in doc:
        for term in query_terms:
            for inst in page.search_for(term, quads=True):
                page.add_highlight_annot(inst.rect)
    highlighted_path = pdf_path.replace(".pdf", "_highlighted.pdf")
    doc.save(highlighted_path)
    return highlighted_path

def debug_quote_processing(quotes_dict, task_id):
    """Debug function to trace quote processing"""
    logging.info(f"[DEBUG {task_id}] === QUOTE PROCESSING DEBUG ===")
    
    total_verified = 0
    total_unverified = 0
    
    for key, quote_info in quotes_dict.items():
        verified_quotes = quote_info.get('quotes', [])
        unverified_quotes = quote_info.get('unverified_quotes', [])
        
        logging.info(f"[DEBUG {task_id}] Source {key}:")
        logging.info(f"[DEBUG {task_id}]   Verified quotes: {len(verified_quotes)}")
        logging.info(f"[DEBUG {task_id}]   Unverified quotes: {len(unverified_quotes)}")
        
        for i, quote_tuple in enumerate(verified_quotes):
            logging.info(f"[DEBUG {task_id}]     V{i}: {len(quote_tuple)} elements - {quote_tuple[1][:50]}...")
            total_verified += 1
            
        for i, quote_tuple in enumerate(unverified_quotes):
            logging.info(f"[DEBUG {task_id}]     U{i}: {len(quote_tuple)} elements - {quote_tuple[1][:50]}...")
            total_unverified += 1
    
    logging.info(f"[DEBUG {task_id}] TOTALS: {total_verified} verified, {total_unverified} unverified")
    return total_verified, total_unverified

def classify_quotes(quotes_dict, file_directory):
    """
    Step 4 of the Pipeline:
    -----------------------
    This function classifies the extracted quotes into predefined categories.
    Now includes both verified and unverified quotes with verification status.
    
    Params:
        quotes_dict (dict): A dictionary containing the previously extracted quotes.
        file_directory (str): The path to the directory containing the input data and the result folder.
    Returns:
        pd.DataFrame : A pandas DF containing the classified quotes.
    """
    try:
        app.logger.info(f"Starting classification with {len(quotes_dict)} quotes")
        
        # DEBUG: Check what we're starting with
        task_id = processing_tasks.get('current_task_id', 'unknown')
        debug_quote_processing(quotes_dict, task_id)
        
        # Prepare quotes_dict to include both verified and unverified quotes
        enhanced_quotes_dict = {}
        total_verified = 0
        total_unverified = 0

        for key, quote_info in quotes_dict.items():
            enhanced_quotes_dict[key] = quote_info.copy()
            
            # Combine verified and unverified quotes with verification status
            all_quotes = []

            # Add verified quotes (should have 3 elements: index, text, verified)
            verified_quotes = quote_info.get('quotes', [])
            for quote_tuple in verified_quotes:
                if len(quote_tuple) >= 3:
                    # New format with verification status
                    all_quotes.append((quote_tuple[0], quote_tuple[1], quote_tuple[2]))
                    if quote_tuple[2]:  # if verified
                        total_verified += 1
                    else:
                        total_unverified += 1
                elif len(quote_tuple) >= 2:
                    # Old format without verification status - assume verified
                    all_quotes.append((quote_tuple[0], quote_tuple[1], True))
                    total_verified += 1           
                else:
                    app.logger.warning(f"[CLASSIFY {task_id}] Invalid verified quote tuple: {quote_tuple}")
            
            # Add unverified quotes  
            unverified_quotes = quote_info.get('unverified_quotes', [])
            for quote_tuple in unverified_quotes:
                if len(quote_tuple) >= 3:
                    all_quotes.append((quote_tuple[0], quote_tuple[1], quote_tuple[2]))
                    if quote_tuple[2]:
                        total_verified += 1
                    else:
                        total_unverified += 1
                elif len(quote_tuple) >= 2:
                    # Old format - assume unverified
                    all_quotes.append((quote_tuple[0], quote_tuple[1], False))
                    total_unverified += 1
                else:
                    app.logger.warning(f"[CLASSIFY {task_id}] Invalid unverified quote tuple: {quote_tuple}")
            
            # Replace the quotes list with enhanced version - Keep the 3-element structure for quotes
            enhanced_quotes_dict[key]['quotes'] = all_quotes
            # Remove the separate unverified_quotes since they're now combined
            enhanced_quotes_dict[key].pop('unverified_quotes', None)

        app.logger.info(f"Processing {total_verified} verified + {total_unverified} unverified quotes = {total_verified + total_unverified} total")

        # Classification of quotes into predefined categories
        output_df = tagging_classifier_quotes(quotes_dict=enhanced_quotes_dict, llm=LLM, fewshot=True)
        
        # Verification status is already included in the DataFrame
        if 'verification_status' in output_df.columns:
            verified_count = (output_df['verification_status'] == 'Verified').sum()
            unverified_count = (output_df['verification_status'] == 'Unverified').sum()
            app.logger.info(f"[CLASSIFY {task_id}] Final verification counts: {verified_count} verified, {unverified_count} unverified")

        # Save results in Excel and CSV{LLM.model_name}_{namespace}
        by_products_folder = os.path.join(get_task_results_folder(task_id), 'by-products')
        os.makedirs(by_products_folder, exist_ok=True)  # Ensure the subfolder exists
        
        try:
            excel_path = os.path.join(by_products_folder, f"{os.path.splitext(os.path.basename(file_directory))[0]}_fewshot-tagging_{LLM.model_name}.xlsx")
            output_df.to_excel(excel_path)
            app.logger.info(f"[CLASSIFY {task_id}] Saved Excel: {excel_path}")
        except Exception as excel_error:
            app.logger.error(f"[CLASSIFY {task_id}] Failed to save Excel: {excel_error}")
            
        try:
            csv_path = os.path.join(by_products_folder, f"{os.path.splitext(os.path.basename(file_directory))[0]}_fewshot-tagging_{LLM.model_name}.csv")
            output_df.to_csv(csv_path)
            app.logger.info(f"[CLASSIFY {task_id}] Saved CSV: {csv_path}")
        except Exception as csv_error:
            app.logger.error(f"[CLASSIFY {task_id}] Failed to save CSV: {csv_error}")


        # TODO: Save the classified quotes to a JSON file
    #    classified_quotes_path = os.path.join(by_products_folder, f"{os.path.splitext(os.path.basename(file_directory))[0]}_classified-quotes.json")
    #    with open(classified_quotes_path, 'w', encoding='utf-8') as f:
    #        json.dump(classified_quotes, f, ensure_ascii=False, indent=4)

        return output_df
    
    except Exception as e:
        app.logger.error(f"Classification failed: {e}")
        app.logger.error(f"Classification traceback: {traceback.format_exc()}")
        raise  # Re-raise to be caught by process_document

def postprocess_results(file_directory, output_df, filename, task_id, summary_data=None):
    """
    Post-processing of Results:
    -----------------------
    This function performs several postprocessing steps on the output DataFrame from the pipeline.
    It generates highlighted PDFs, and saves specific subsets of the data (targets, mitigation measures, 
    and adaptation measures) to both Excel and CSV files, and includes verification status.

    Params:
        input_file (str): The directory of the input file.
        output_df (pd.DataFrame): The DataFrame containing the results of the pipeline.
        filename (str): The name of the file.
        task_id (str): Unique task identifier.
        summary_data (dict): The summary data generated by the AI model.
    Returns:
        None
    """

    # create task-specific output folder
    output_folder = os.path.join(get_task_results_folder(task_id), 'output')
    os.makedirs(output_folder, exist_ok=True)  # Ensure the subfolder exists

    # highlight quotes
    create_highlighted_pdf(file_directory, 
                            quotes=output_df['quote'],
                            output_path=os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_highlighted.pdf"),
                            df = output_df
                        )

    #  targets
    targets_output_df = output_df[output_df['target']=='True']
    targets_output_df = targets_output_df[['quote', 'page', 'target_labels', 'target_area', 'ghg_target', 'conditionality', 'verification_status']]
    targets_output_df = targets_output_df.rename(columns={
        'quote': 'Content',
        'page': 'Page Number',
        'target_area': 'Target area',
        'ghg_target': 'GHG target?',
        'conditionality': 'Conditionality',
        'verification_status': 'Verification Status'
    })
    targets_output_df["Target scope"] = ""
    targets_output_df["Target type"] = ""
    targets_output_df["Target year"] = ""
    targets_output_df = targets_output_df[["Target area", "Target scope", "GHG target?", "Target type", "Conditionality", "Target year", "Content", "Page Number", "Verification Status"]]

    #  mitigation
    mitigation_output_df = output_df[output_df['mitigation_measure']=='True']
    mitigation_output_df = mitigation_output_df[['quote', 'page', 'measure_labels', 'category', 'purpose', 'instrument', 'asi', 'verification_status']]
    mitigation_output_df = mitigation_output_df.rename(columns={
        'quote': 'Quote',
        'page': 'Page Number',
        'category': 'Category',
        'purpose': 'Purpose',
        'instrument': 'Instrument',
        'asi': 'A-S-I',
        'verification_status': 'Verification Status'
    })
    mitigation_output_df = mitigation_output_df[["Category", "Purpose", "Instrument", "Quote", "A-S-I", "Page Number", "Verification Status"]]

    #  adaptation
    adaptation_output_df = output_df[output_df['adaptation_measure']=='True']
    adaptation_output_df = adaptation_output_df[['quote', 'page', 'measure_labels', 'category', 'instrument', 'verification_status']]
    adaptation_output_df = adaptation_output_df.rename(columns={
        'quote': 'Quote',
        'page': 'Page Number',
        'category': 'Category',
        'instrument': 'Measure',
        'verification_status': 'Verification Status'
    })
    adaptation_output_df = adaptation_output_df[["Category", "Measure", "Quote", "Page Number", "Verification Status"]]

    # Save summary as human-readable text to output folder
    summary_txt_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_ai_summary.txt")
    if summary_data:
        with open(summary_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"TRANSPORT POLICY ANALYSIS - AI SUMMARY\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Document: {summary_data['statistics']['document_name']}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Quotes Analyzed: {summary_data['statistics']['total_quotes']}\n")
            f.write(f"Tokens Used for AI Summary: {summary_data['tokens_used']}\n\n")
            
            f.write(f"CLASSIFICATION BREAKDOWN:\n")
            f.write(f"{'-'*25}\n")
            f.write(f"🎯 Targets: {summary_data['statistics']['targets']} quotes\n")
            f.write(f"🛡️ Mitigation: {summary_data['statistics']['mitigation']} quotes\n")
            f.write(f"🌿 Adaptation: {summary_data['statistics']['adaptation']} quotes\n\n")
            ''''
            # Write structured sections if available
            sections = summary_data.get('summary_sections', {})
            
            if sections.get('executive_summary'):
                f.write(f"EXECUTIVE SUMMARY:\n")
                f.write(f"{'-'*18}\n")
                f.write(f"{sections['executive_summary']}\n\n")
            
            if sections.get('key_findings'):
                f.write(f"KEY FINDINGS:\n")
                f.write(f"{'-'*13}\n")
                for i, finding in enumerate(sections['key_findings'], 1):
                    f.write(f"{i}. {finding}\n")
                f.write(f"\n")
            
            # LEGACY
            if sections.get('policy_implications'):
                f.write(f"POLICY IMPLICATIONS:\n")
                f.write(f"{'-'*20}\n")
                f.write(f"{sections['policy_implications']}\n\n")
            
            # LEGACY
            if sections.get('notable_patterns'):
                f.write(f"NOTABLE PATTERNS:\n")
                f.write(f"{'-'*17}\n")
                f.write(f"{sections['notable_patterns']}\n\n")
            '''
            # Raw summary
            if summary_data.get('summary_text'):
                f.write(f"AI ANALYSIS:\n")
                f.write(f"{'-'*12}\n")
                f.write(f"{summary_data['summary_text']}\n\n")
            
            f.write(f"\n{'='*50}\n")
            f.write(f"Generated by Transport Policy Miner\n")
            f.write(f"AI Model: Azure OpenAI GPT-4o mini\n")
        
        logging.info(f"AI summary saved to output folder: {summary_txt_path}")

    else:
        with open(summary_txt_path, 'w', encoding='utf-8') as f:
            f.write("No summary data available.\n")
        logging.info(f"No summary data provided; created empty summary file at: {summary_txt_path}")

    # Save to Excel with conditional formatting
    excel_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_results.xlsx")
    
    try:
        # Create Excel writer with xlsxwriter engine
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            # Write dataframes to sheets
            targets_output_df.to_excel(writer, sheet_name="Targets", index=False)
            mitigation_output_df.to_excel(writer, sheet_name="Mitigation", index=False)
            adaptation_output_df.to_excel(writer, sheet_name="Adaptation", index=False)

            # ✅ FIXED: Access workbook AFTER writing data
            workbook = writer.book
            
            # Define formats for verification status
            verified_format = workbook.add_format({
                'bg_color': '#d4edda',
                'font_color': '#155724',
                'bold': True
            })
            
            unverified_format = workbook.add_format({
                'bg_color': '#f8d7da',
                'font_color': '#721c24',
                'bold': True
            })
            
            # Apply formatting to each sheet
            sheet_configs = [
                ('Targets', targets_output_df, writer.sheets['Targets']),
                ('Mitigation', mitigation_output_df, writer.sheets['Mitigation']),
                ('Adaptation', adaptation_output_df, writer.sheets['Adaptation'])
            ]
            
            for sheet_name, df, worksheet in sheet_configs:
                if len(df) == 0:
                    continue
                    
                # Find verification status column (should be last)
                col_idx = len(df.columns) - 1
                col_letter = chr(65 + col_idx)  # A, B, C, etc.
                last_row = len(df) + 1  # +1 for header
                
                # Apply conditional formatting for "Unverified" (red)
                worksheet.conditional_format(f'{col_letter}2:{col_letter}{last_row}', {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': 'Unverified',
                    'format': unverified_format
                })
                
                # Apply conditional formatting for "Verified" (green)
                worksheet.conditional_format(f'{col_letter}2:{col_letter}{last_row}', {
                    'type': 'text',
                    'criteria': 'containing',
                    'value': 'Verified',
                    'format': verified_format
                })
                
                # Auto-adjust column widths
                for i, col in enumerate(df.columns):
                    max_len = max(
                        df[col].astype(str).apply(len).max(),
                        len(str(col))
                    ) + 2
                    worksheet.set_column(i, i, min(max_len, 50))  # Cap at 50
        
        logging.info(f"Excel file with formatting saved: {excel_path}")
        
    except Exception as e:
        logging.error(f"Error creating Excel with formatting: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        
        # Fallback: Save without formatting
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                targets_output_df.to_excel(writer, sheet_name="Targets", index=False)
                mitigation_output_df.to_excel(writer, sheet_name="Mitigation", index=False)
                adaptation_output_df.to_excel(writer, sheet_name="Adaptation", index=False)
            logging.info(f"Excel saved without conditional formatting (fallback)")
        except Exception as fallback_error:
            logging.error(f"Fallback Excel save also failed: {fallback_error}")
            raise

    logging.info(f"Post-processing completed for: {filename}")


# Default provided - DEPRECARED
def classify_citations(citations):
    results = []
    for item in citations:
        prompt = (
            f"You are a climate policy expert. Classify the following citation into one of the following categories: {', '.join(TAXONOMY)}.\n"
            f"Return a JSON object with 'label' and 'score' (confidence 0.0-1.0).\n\n"
            f"Citation: \"{item['text']}\""
        )

        response = LLM.invoke([HumanMessage(content=prompt)])

        try:
            result = json.loads(response.content)
            item["label"] = result.get("label", "Unknown")
            item["confidence"] = result.get("confidence", 0.0)
        except Exception:
            item["label"] = "Error"
            item["confidence"] = 0.0
        results.append(item)
    return results

def create_results_zip(task_id):
    """
    Creates a zip file containing the entire results folder
    
    Args:
        task_id (str): Unique task identifier

    Returns:
        io.BytesIO: An in-memory zip file containing the results folder
    """
    task_results_folder = get_task_results_folder(task_id)
    
    # Check if the results folder exists and contains files
    if not os.path.exists(task_results_folder) or not any(os.scandir(task_results_folder)):
        raise FileNotFoundError(f"The results folder is empty or does not exist for task {task_id}.")
    
    memory_file = io.BytesIO()
    
    with ZipFile(memory_file, 'w') as zf:
        for root, dirs, files in os.walk(task_results_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate the relative path for the file in the zip
                relative_path = os.path.relpath(file_path, task_results_folder)
                zf.write(file_path, relative_path)

        # Add the log file to the zip
        log_file = os.path.join(task_results_folder, 'logs', 'app.log')
        if os.path.exists(log_file):
            zf.write(log_file, os.path.relpath(log_file, task_results_folder))

        # Create a manifest file listing the contents of the zip
        manifest_content = f"""TRANSPORT POLICY MINER - RESULTS PACKAGE
==========================================
Task ID: {task_id}
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

CONTENTS:
- output/: Final analysis results (Excel, highlighted PDF, AI summary)
- by-products/: Intermediate processing files (JSON, CSV)
- vector_stores/: Semantic search indexes
- logs/: Processing logs

Each task has its own isolated folder to prevent conflicts.
"""
        zf.writestr("README.txt", manifest_content)

    # Move to the beginning of the BytesIO buffer
    memory_file.seek(0)
    return memory_file

# DEPRECATED: replaced by cleanup_old_task_results() 
def cleanup_old_results(keep_base_filename=None):
    """
    Clean up old result files, optionally keeping files for a specific document
    
    Args:
        keep_base_filename (str): Base filename to keep (others will be deleted)
    """
    try:
        results_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        
        if not os.path.exists(results_folder):
            return
        
        files_deleted = 0
        folders_to_check = ['output', 'by-products', 'vector_stores']
        
        for folder_name in folders_to_check:
            folder_path = os.path.join(results_folder, folder_name)
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    
                    # Delete files that don't belong to the current document
                    if keep_base_filename:
                        if keep_base_filename.lower() not in file.lower():
                            try:
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                                    files_deleted += 1
                                elif os.path.isdir(file_path):
                                    import shutil
                                    shutil.rmtree(file_path)
                                    files_deleted += 1
                            except Exception as e:
                                app.logger.warning(f"Could not delete {file_path}: {e}")
        
        if files_deleted > 0:
            app.logger.info(f"Cleaned up {files_deleted} old result files")
            
    except Exception as e:
        app.logger.error(f"Error during cleanup: {e}")

def create_partial_results_zip(task_id):
    """
    Creates a zip file containing any available results, even if processing failed
    
    Args:
        task_id (str): Unique task identifier
    Returns:
        io.BytesIO: An in-memory zip file containing available results
    """
    task_results_folder = get_task_results_folder(task_id)

    # Check if the results folder exists
    if not os.path.exists(task_results_folder):
        raise FileNotFoundError(f"No results folder found for task {task_id}.")
    
    memory_file = io.BytesIO()
    files_added = 0
    
    with ZipFile(memory_file, 'w') as zf:
        # Add any files that exist in the results folder structure
        for root, dirs, files in os.walk(task_results_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate the relative path for the file in the zip
                relative_path = os.path.relpath(file_path, task_results_folder)
                zf.write(file_path, relative_path)
                files_added += 1
        
        # If no files were found, add a readme explaining the situation
        if files_added == 0:
            readme_content = """PARTIAL RESULTS README
============================
Task ID: {task_id}

This download contains partial results from a processing task that encountered an error.

Unfortunately, no output files were generated before the error occurred.
However, you may find useful information in the error details provided in the web interface.

For support, please provide:
1. The task ID: {task_id}
2. The error message and traceback
3. The original document you were trying to process

"""
            zf.writestr("README_PARTIAL_RESULTS.txt", readme_content)

    # Move to the beginning of the BytesIO buffer
    memory_file.seek(0)
    return memory_file

def generate_summary_insights(LLM, citations, classified_quotes, original_filename):
    """Generate AI summary and insights based on extracted quotes"""
    try:
        # Prepare data for summary
        verified_quotes = sum(len(item.get('quotes', [])) for item in citations.values())
        unverified_quotes = sum(len(item.get('unverified_quotes', [])) for item in citations.values())
        total_quotes = verified_quotes + unverified_quotes

        # Count classifications
        classification_counts = {
            'targets': 0,
            'mitigation': 0,
            'adaptation': 0
        }
        
        # Collect sample quotes by category
        categorized_quotes = {
            'economy_wide_targets': [],
            'transport_targets': [],
            'mitigation_actions': [],
            'adaptation_actions': []
        }
        
        for quote_data in classified_quotes:
            quote_text = quote_data.get('quote', '')
            
            # Count for statistics
            if quote_data.get('target') == 'True':
                classification_counts['targets'] += 1
            if quote_data.get('mitigation_measure') == 'True':
                classification_counts['mitigation'] += 1
            if quote_data.get('adaptation_measure') == 'True':
                classification_counts['adaptation'] += 1

            # Categorize quotes for targeted summaries
            # Economy-wide targets (net-zero, economy-wide mitigation)
            if (quote_data.get('target') == 'True' and 
                (quote_data.get('sector_level') == 'False' or quote_data.get('net_zero') == 'True')):
                categorized_quotes['economy_wide_targets'].append(quote_text)
            
            # Transport targets (transport sector specific targets)
            elif (quote_data.get('target') == 'True' and quote_data.get('transport') == 'True'):
                categorized_quotes['transport_targets'].append(quote_text)
            
            # Mitigation actions
            if quote_data.get('mitigation_measure') == 'True':
                categorized_quotes['mitigation_actions'].append(quote_text)
            
            # Adaptation actions
            if quote_data.get('adaptation_measure') == 'True':
                categorized_quotes['adaptation_actions'].append(quote_text)
        
        # Create prompt for AI summary
        prompt = f"""
You are a climate policy analyst, specialised in transport policy. Please analyze the following transport policy document analysis results and provide a comprehensive summary with targeted key insights.

Document: {original_filename}
Total Quotes Extracted: {total_quotes}

Classification Results:
- Targets: {classification_counts['targets']} quotes
- Mitigation Measures: {classification_counts['mitigation']} quotes  
- Adaptation Measures: {classification_counts['adaptation']} quotes

ECONOMY-WIDE TARGETS (Found: {len(categorized_quotes['economy_wide_targets'])} quotes):
{chr(10).join(f"- {quote}" for quote in categorized_quotes['economy_wide_targets'])}

TRANSPORT TARGETS (Found: {len(categorized_quotes['transport_targets'])} quotes):
{chr(10).join(f"- {quote}" for quote in categorized_quotes['transport_targets'])}

MITIGATION ACTIONS (Found: {len(categorized_quotes['mitigation_actions'])} quotes):
{chr(10).join(f"- {quote}" for quote in categorized_quotes['mitigation_actions'])}

ADAPTATION ACTIONS (Found: {len(categorized_quotes['adaptation_actions'])} quotes):
{chr(10).join(f"- {quote}" for quote in categorized_quotes['adaptation_actions'])}

Please provide:
1. Executive Summary (2-3 sentences about the document's main transport policy focus)

2. Key Findings (3-4 bullet points about the most important insights across all categories)

3. Economy-wide Targets Summary (analyze the economy-wide and net-zero targets):
   - Each bullet point should start with an ACTION VERB (e.g., "Commits to...", "Establishes...", "Targets...")
   - Focus on what the country plans to DO, not just reproduce quotes
   - Convert any CO2 equivalent units to megatonnes (MT CO2eq) for consistency
   - If no economy-wide targets found, state "No specific economy-wide targets identified" and nothing else

4. Transport Targets Summary (analyze transport sector specific targets):
   - Each bullet point should start with an ACTION VERB (e.g., "Aims to...", "Plans to...", "Seeks to...")
   - Focus on specific transport sector commitments and goals
   - Convert any CO2 equivalent units to megatonnes (MT CO2eq) for consistency
   - If no transport targets found, state "No specific transport sector targets identified" and nothing else

5. Mitigation Actions Summary (analyze planned mitigation measures):
   - Each bullet point should start with an ACTION VERB (e.g., "Implements...", "Develops...", "Promotes...")
   - Focus on concrete actions and policies being implemented
   - Group similar measures together where possible
   - If no mitigation actions found, state "No specific mitigation actions identified" and nothing else

6. Adaptation Actions Summary (analyze planned adaptation measures):
   - Each bullet point should start with an ACTION VERB (e.g., "Strengthens...", "Builds...", "Enhances...")
   - Focus on resilience-building and adaptation measures
   - If no adaptation actions found, state "No specific adaptation actions identified" and nothing else

IMPORTANT INSTRUCTIONS:
- Always convert CO2 equivalent numbers to megatonnes (MT CO2eq) - e.g., "1000 kt CO2eq" becomes "1 MT CO2eq"
- Each summary bullet should describe what the country IS DOING or PLANS TO DO
- Do not simply reproduce quotes - synthesize and summarize the content
- Use clear, policy-relevant language suitable for decision makers
- Structure the response into clear sections as outlined above
- Do NOT USE ANY FORMATTING except for the section titles

Keep the response concise but informative, suitable for policy makers and researchers.
"""
        # ✅ NEW: Save the prompt to by-products folder for transparency
        task_id = processing_tasks.get('current_task_id', 'unknown')
        by_products_folder = os.path.join(get_task_results_folder(task_id), 'by-products')
        os.makedirs(by_products_folder, exist_ok=True)
        
        base_filename = os.path.splitext(original_filename)[0]
        prompt_path = os.path.join(by_products_folder, f"{base_filename}_ai_summary_prompt.txt")
        
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("AI SUMMARY GENERATION PROMPT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Document: {original_filename}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {LLM.model_name}\n\n")
            f.write("="*80 + "\n")
            f.write("PROMPT SENT TO AI MODEL:\n")
            f.write("="*80 + "\n\n")
            f.write(prompt)
            f.write("\n\n" + "="*80 + "\n")
            f.write("END OF PROMPT\n")
            f.write("="*80 + "\n")
        
        logging.info(f"AI summary prompt saved to: {prompt_path}")

        # Generate summary using your text model
        with get_openai_callback() as cb:
            response = LLM.invoke([HumanMessage(content=prompt)])
            tokens_used = cb.total_tokens
            summary_text = response.content
        
        # Parse the response into structured sections
        summary_sections = parse_summary_response_enhanced(summary_text)

        return {
            'summary_text': summary_text,
            'summary_sections': summary_sections,
            'tokens_used': tokens_used,
            'statistics': {
                'total_quotes': total_quotes,
                'targets': classification_counts['targets'],
                'mitigation': classification_counts['mitigation'],
                'adaptation': classification_counts['adaptation'],
                'document_name': original_filename,
                'categorized_counts': {
                    'economy_wide_targets': len(categorized_quotes['economy_wide_targets']),
                    'transport_targets': len(categorized_quotes['transport_targets']),
                    'mitigation_actions': len(categorized_quotes['mitigation_actions']),
                    'adaptation_actions': len(categorized_quotes['adaptation_actions'])
                }
            }
        }
        
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return {
            'summary_text': f"Summary generation failed: {str(e)}",
            'summary_sections': {},
            'tokens_used': 0,
            'statistics': {
                'total_quotes': 0,
                'targets': 0,
                'mitigation': 0,
                'adaptation': 0,
                'document_name': original_filename,
                'categorized_counts': {
                    'economy_wide_targets': 0,
                    'transport_targets': 0,
                    'mitigation_actions': 0,
                    'adaptation_actions': 0
                }
            }
        }

def parse_summary_response_enhanced(summary_text):
    """Parse AI response into structured sections"""
    sections = {
        'executive_summary': '',
        'key_findings': [],
        'economy_wide_targets': [],
        'transport_targets': [],
        'mitigation_actions': [],
        'adaptation_actions': []
    }
    
    try:
        lines = summary_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            if 'executive summary' in line.lower():
                current_section = 'executive_summary'
                continue
            elif 'key findings' in line.lower():
                current_section = 'key_findings'
                continue
            elif 'economy-wide targets' in line.lower() or 'economy wide targets' in line.lower():
                current_section = 'economy_wide_targets'
                continue
            elif 'transport targets' in line.lower():
                current_section = 'transport_targets'
                continue
            elif 'mitigation actions' in line.lower():
                current_section = 'mitigation_actions'
                continue
            elif 'adaptation actions' in line.lower():
                current_section = 'adaptation_actions'
                continue
            
            # Add content to current section
            if current_section in ['key_findings', 'economy_wide_targets', 'transport_targets', 'mitigation_actions', 'adaptation_actions']:
                if line.startswith(('- ', '• ', '* ', '1.', '2.', '3.', '4.', '5.')):
                    # Remove bullet/number formatting
                    clean_line = line.lstrip('- •*123456789. ')
                    if clean_line:
                        sections[current_section].append(clean_line)
            elif current_section == 'executive_summary':
                if sections[current_section]:
                    sections[current_section] += ' ' + line
                else:
                    sections[current_section] = line
        
    except Exception as e:
        logging.error(f"Error parsing summary: {e}")
    
    return sections


def process_document(task_id, file_path, query_terms, filename, document_pages=None, estimated_minutes=None):
    """
    Process a document in the background and update progress.
    
    Args:
        task_id (str): Unique ID for this processing task
        file_path (str): Path to the PDF file
        query_terms (list): Terms to search for
        filename (str): Original filename
        document_pages (int): Pre-calculated page count
        estimated_minutes (int): Pre-calculated estimated processing time
    """
    task_logger = None  # Initialize to None for error handling
    task_handler = None

    try:
        # Store current task ID for debugging
        processing_tasks['current_task_id'] = task_id

        # Clean up old results before starting new processing
        #base_filename = os.path.splitext(filename)[0]
        #cleanup_old_results(keep_base_filename=base_filename)

        # Create task-specific folder structure
        task_results_folder = get_task_results_folder(task_id)
        by_products_folder = os.path.join(task_results_folder, 'by-products')
        output_folder = os.path.join(task_results_folder, 'output')
        vector_store_folder = os.path.join(task_results_folder, 'vector_stores')
        logs_folder = os.path.join(task_results_folder, 'logs')

        for folder in [by_products_folder, output_folder, vector_store_folder, logs_folder]:
            os.makedirs(folder, exist_ok=True)
        
        # Create task-specific log file
        task_log_file = os.path.join(logs_folder, f'{task_id}_processing.log')
        task_handler = logging.FileHandler(task_log_file)
        task_handler.setLevel(logging.INFO)
        task_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Add task handler to logger
        task_logger = logging.getLogger(f'task_{task_id}')
        task_logger.addHandler(task_handler)
        task_logger.setLevel(logging.INFO)

        # RICH HEADER LOGGING
        task_logger.info("="*80)
        task_logger.info("TRANSPORT POLICY MINER - PROCESSING LOG")
        task_logger.info("="*80)
        task_logger.info(f"Task ID: {task_id}")
        task_logger.info(f"Document: {filename}")
        task_logger.info(f"Document Pages: {document_pages}")
        task_logger.info(f"Estimated Processing Time: {estimated_minutes} minutes")
        task_logger.info(f"Query Terms: {len(query_terms)} terms provided")
        task_logger.info(f"Results Folder: {task_results_folder}")
        task_logger.info(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        task_logger.info("="*80)

        # Initialize progress tracking
        processing_tasks[task_id] = {
            "progress": 0,
            "status": "Starting document analysis...",
            "result": None,
            "error": None,
            "traceback": None,
            "original_filename": filename,  # Store the original filename
            "partial_results": {},  # Store partial results,
            "last_heartbeat": time.time(),  # Add heartbeat timestamp
            "estimated_time_minutes": estimated_minutes,
            "document_pages": document_pages,
            "results_folder": task_results_folder,  # Store the results folder path
            "log_file": task_log_file  # Store the log file path
        }
        
        # Start total runtime timer
        total_start_time = time.time()
        total_tokens_used = 0  # Initialize total token counter
        
        task_logger.info("")
        task_logger.info("PHASE 0: INITIALIZATION")
        task_logger.info("-"*40)

        # Step 0: Initialize task
        # Detect document length for time estimation
        processing_tasks[task_id]["status"] = "Document analyzed: {} pages. Starting text extraction...".format(document_pages)
        processing_tasks[task_id]["progress"] = 5
        processing_tasks[task_id]["last_heartbeat"] = time.time()  
        
        task_logger.info(f"Task initialized successfully")
        task_logger.info(f"Folder structure created")
        task_logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")

        # Step 1: Extract text with Unstructured (20%)
        task_logger.info("")
        task_logger.info("PHASE 1: TEXT EXTRACTION")
        task_logger.info("-"*40)
        step_start_time = time.time()
        processing_tasks[task_id]["status"] = "Extracting text from document..."
        task_logger.info(f"Starting text extraction from PDF: {file_path}")
        task_logger.info(f"Using Unstructured library with multi-language support")

        unstructured_text = extract_text_with_unstructured(file_path, task_id)
        
        preprocessing_time = time.time() - step_start_time
        processing_tasks[task_id]["progress"] = 20
        processing_tasks[task_id]["status"] = "Text extraction completed."
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        processing_tasks[task_id]["partial_results"]["text_extraction"] = True  # Mark as completed
        processing_tasks[task_id]["partial_results"]["text_partitions"] = unstructured_text[:50]  # Limit for performance
        processing_tasks[task_id]["partial_results"]["total_partitions"] = len(unstructured_text)
        processing_tasks[task_id]["partial_results"]["text_extraction_complete"] = True
        processing_tasks[task_id]["status"] = "Text extraction completed."

        task_logger.info(f"Text extraction completed in {preprocessing_time:.2f} seconds")
        task_logger.info(f"Extracted {len(unstructured_text)} text chunks/partitions")
        task_logger.info(f"Average chunk length: {sum(len(chunk.get('text', '')) for chunk in unstructured_text) / len(unstructured_text):.0f} characters")
        task_logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        
        # Memory cleanup
        gc.collect()

        # Step 2: Perform semantic search (40%)
        task_logger.info("")
        task_logger.info("PHASE 2: SEMANTIC SEARCH")
        task_logger.info("-"*40)
        step_start_time = time.time()
        processing_tasks[task_id]["status"] = "Preparing semantic search..."
        processing_tasks[task_id]["progress"] = 25
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        
        task_logger.info(f"Starting semantic search with FAISS vector store")
        task_logger.info(f"Using Azure OpenAI embeddings: {os.environ.get('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME')}")
        task_logger.info(f"Searching with {len(query_terms)} query terms")
        task_logger.info(f"Creating vector store at: {vector_store_folder}")

        processing_tasks[task_id]["status"] = "Performing semantic search..."
        processing_tasks[task_id]["last_heartbeat"] = time.time()

        chunks = semantic_search_FAISS(unstructured_text, query_terms, 
                                      os.path.splitext(os.path.basename(file_path))[0],
                                      task_id=task_id,
                                      embedding_model=EMBEDDING)

        retrieval_time = time.time() - step_start_time
        processing_tasks[task_id]["partial_results"]["search_results"] = chunks[:20]  # Limit for performance
        processing_tasks[task_id]["partial_results"]["total_search_results"] = len(chunks)
        processing_tasks[task_id]["partial_results"]["semantic_search_complete"] = True  # Mark as completed
        processing_tasks[task_id]["progress"] = 30
        processing_tasks[task_id]["status"] = "Semantic search completed."
        processing_tasks[task_id]["last_heartbeat"] = time.time()

        task_logger.info(f"Semantic search completed in {retrieval_time:.2f} seconds")
        task_logger.info(f"Found {len(chunks)} relevant text chunks")
        task_logger.info(f"Average similarity score: {sum(chunk.get('score', 0) for chunk in chunks) / len(chunks):.3f}")
        task_logger.info(f"Score range: {min(chunk.get('score', 0) for chunk in chunks):.3f} - {max(chunk.get('score', 0) for chunk in chunks):.3f}")
        task_logger.info(f"Vector store saved successfully")
        task_logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        
        # Memory cleanup
        gc.collect()

        # Step 3: Extract quotes (60%)
        task_logger.info("")
        task_logger.info("PHASE 3: QUOTE EXTRACTION")
        task_logger.info("-"*40)
        step_start_time = time.time()
        processing_tasks[task_id]["status"] = "Extracting quotes from relevant sections..."
        processing_tasks[task_id]["progress"] = 40
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        
        estimated_api_calls = len(chunks) * 2
        task_logger.info(f"Starting quote extraction with Azure OpenAI")
        task_logger.info(f"Using model: {os.environ.get('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')}")
        task_logger.info(f"Processing {len(chunks)} text chunks")
        task_logger.info(f"Estimated API calls: {estimated_api_calls}")
        
        if estimated_api_calls > 50:
            task_logger.warning(f"⚠️  High API call count may trigger rate limits")
        
        citations, tokens_used = extract_quotes(chunks, os.path.basename(file_path), task_id)
        total_tokens_used += tokens_used  # Update total tokens used

        
        quotation_time = time.time() - step_start_time        
        processing_tasks[task_id]["partial_results"]["extracted_quotes"] = citations
        processing_tasks[task_id]["partial_results"]["quote_extraction_complete"] = True
        processing_tasks[task_id]["progress"] = 60
        processing_tasks[task_id]["status"] = "Quote extraction completed."
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        processing_tasks[task_id]["partial_results"]["quote_extraction"] = True  # Mark as completed

        # Count verified vs unverified quotes
        verified_count = sum(len(item.get('quotes', [])) for item in citations.values())
        unverified_count = sum(len(item.get('unverified_quotes', [])) for item in citations.values())

        task_logger.info(f"Quote extraction completed in {quotation_time:.2f} seconds")
        task_logger.info(f"Extracted {verified_count} verified quotes")
        task_logger.info(f"Extracted {unverified_count} unverified quotes")
        task_logger.info(f"Total quotes: {verified_count + unverified_count}")
        task_logger.info(f"Tokens used: {tokens_used:,}")
        task_logger.info(f"API calls per second: {estimated_api_calls/quotation_time:.2f}")
        task_logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        
        # Memory cleanup
        gc.collect()

        # Step 4: Classify quotes (80%)
        task_logger.info("")
        task_logger.info("PHASE 4: QUOTE CLASSIFICATION")
        task_logger.info("-"*40)
        step_start_time = time.time()
        processing_tasks[task_id]["status"] = "Preparing AI classification..."
        processing_tasks[task_id]["progress"] = 65
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        
        task_logger.info(f"Starting quote classification with few-shot learning")
        task_logger.info(f"Using model: {LLM.model_name}")
        task_logger.info(f"Classifying {verified_count + unverified_count} quotes")
        
        processing_tasks[task_id]["status"] = "Classifying extracted quotes..."
        processing_tasks[task_id]["progress"] = 70
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        
        with get_openai_callback() as cb:
            classified = classify_quotes(citations, file_path)
            tokens_used = cb.total_tokens
            total_tokens_used += tokens_used  # Update total tokens used

        task_logger.info(f"Quote classification completed successfully")

        classification_time = time.time() - step_start_time
        processing_tasks[task_id]["partial_results"]["classified_quotes"] = _sanitize_for_json(classified.to_dict('records'))
        processing_tasks[task_id]["partial_results"]["classification_complete"] = True
        processing_tasks[task_id]["progress"] = 80
        processing_tasks[task_id]["status"] = "AI classification completed."
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        processing_tasks[task_id]["partial_results"]["classification"] = True  # Mark as completed
        
        # Count classifications
        targets_count = (classified['target'] == 'True').sum()
        mitigation_count = (classified['mitigation_measure'] == 'True').sum()
        adaptation_count = (classified['adaptation_measure'] == 'True').sum()

        task_logger.info(f"Classification completed in {classification_time:.2f} seconds")
        task_logger.info(f"Classified {len(classified)} quotes")
        task_logger.info(f"Targets: {targets_count} quotes")
        task_logger.info(f"Mitigation measures: {mitigation_count} quotes")
        task_logger.info(f"Adaptation measures: {adaptation_count} quotes")
        task_logger.info(f"Tokens used: {tokens_used:,}")
        task_logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")

        # Memory cleanup
        gc.collect()

        # Step 5: Post-process results (100%)
        task_logger.info("")
        task_logger.info("PHASE 5: AI SUMMARY GENERATION")
        task_logger.info("-"*40)
        step_start_time = time.time()
        processing_tasks[task_id]["status"] = "Generating AI summary and insights..."
        processing_tasks[task_id]["progress"] = 85
        
        task_logger.info(f"Generating comprehensive AI summary")
        task_logger.info(f"Using model: {LLM.model_name}")

        with get_openai_callback() as cb:
            summary_data = generate_summary_insights(LLM, citations, classified.to_dict('records'), filename)
            tokens_used = cb.total_tokens
            total_tokens_used += tokens_used  # Update total tokens used
        
        processing_tasks[task_id]["summary"] = summary_data # Store summary data
        processing_tasks[task_id]["partial_results"]["summary_complete"] = True
        summary_time = time.time() - step_start_time
        task_logger.info(f"AI summary generated in {summary_time:.2f} seconds")
        task_logger.info(f"Summary sections: {len(summary_data.get('summary_sections', {}))}")
        task_logger.info(f"Tokens used: {tokens_used:,}")
        task_logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")

        processing_tasks[task_id]["status"] = "Finalizing results..."
        processing_tasks[task_id]["progress"] = 90
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        
        task_logger.info(f"Starting post-processing")
        task_logger.info(f"Creating highlighted PDF")
        task_logger.info(f"Generating Excel output with conditional formatting")
        task_logger.info(f"Saving results to: {output_folder}")

        postprocess_results(file_path, classified, filename, task_id, summary_data)
        
        postprocessing_time = time.time() - step_start_time
        processing_tasks[task_id]["progress"] = 100
        processing_tasks[task_id]["status"] = "Analysis complete"
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        processing_tasks[task_id]["partial_results"]["postprocessing"] = True  # Mark as completed

        task_logger.info(f"Post-processing completed in {postprocessing_time:.2f} seconds")
        task_logger.info(f"Highlighted PDF created")
        task_logger.info(f"Excel file with 3 sheets created")
        task_logger.info(f"AI summary text file created")
        task_logger.info(f"All files saved to output folder")

        # Calculate total runtime
        total_elapsed_time = time.time() - total_start_time

        # ✅ FINAL SUMMARY - BEFORE CLEANUP
        task_logger.info("")
        task_logger.info("="*80)
        task_logger.info("PROCESSING COMPLETED SUCCESSFULLY")
        task_logger.info("="*80)
        task_logger.info(f"Total Runtime: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.1f} minutes)")
        task_logger.info(f"Total Tokens Used: {total_tokens_used:,}")
        
        # Calculate token costs (approximate)
        # GPT-4 pricing: ~$0.03 per 1K input tokens, ~$0.06 per 1K output tokens (average ~$0.045)
        estimated_cost = (total_tokens_used / 1000) * 0.045
        task_logger.info(f"Estimated API Cost: ${estimated_cost:.2f} USD")
        
        task_logger.info(f"Performance:")
        task_logger.info(f"  - Tokens per second: {total_tokens_used/total_elapsed_time:.1f}")
        task_logger.info(f"  - Pages per minute: {document_pages/(total_elapsed_time/60):.1f}")
        task_logger.info(f"  - Final memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        
        task_logger.info(f"Output Summary:")
        task_logger.info(f"  - Targets found: {targets_count}")
        task_logger.info(f"  - Mitigation measures: {mitigation_count}")
        task_logger.info(f"  - Adaptation measures: {adaptation_count}")
        task_logger.info(f"  - Total quotes: {verified_count + unverified_count}")
        
        task_logger.info(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        task_logger.info("="*80)

        # Store results for retrieval
        processing_tasks[task_id]["status"] = "Analysis complete"
        processing_tasks[task_id]["result"] = citations
        processing_tasks[task_id]["total_runtime_seconds"] = total_elapsed_time  
        processing_tasks[task_id]["total_tokens_used"] = total_tokens_used    

        # Clean up logger AFTER all logging is done
        if task_handler:
            task_handler.close()
            task_logger.removeHandler(task_handler)
        
    except Exception as e:
        # Capture the full traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_traceback = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # Log error to task-specific log if available
        if task_logger:
            task_logger.error("")
            task_logger.error("="*80)
            task_logger.error("PROCESSING FAILED")
            task_logger.error("="*80)
            task_logger.error(f"Error: {str(e)}")
            task_logger.error(f"Error Type: {type(e).__name__}")
            task_logger.error(f"Traceback:")
            task_logger.error(error_traceback)
            task_logger.error("="*80)
        
        # Handle any errors with detailed information
        processing_tasks[task_id]["error"] = str(e)
        processing_tasks[task_id]["traceback"] = error_traceback
        processing_tasks[task_id]["status"] = f"Error: {str(e)}"
        processing_tasks[task_id]["failed"] = True 
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        logging.error(f"Error processing document: {e}")
        logging.error(f"Traceback: {error_traceback}")
        app.logger.error(f"Task {task_id} failed: {e}")
        
        if 'task_logger' in locals():
            task_logger.error(f"Processing failed: {e}")
            task_handler.close()
            task_logger.removeHandler(task_handler)

@app.route('/health')
def health_check():
    """Health check endpoint for Render.com"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "port": os.environ.get('PORT', 'not set'),
        "service": "climate-policy-miner"
    }), 200

'''Queue
@app.route('/api/release-slot/<task_id>', methods=['POST'])
def api_release_slot(task_id):
    """API endpoint to release queue slot"""
    success = release_queue_slot(task_id)
    return jsonify({"released": success})
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Home route that handles file uploads, URL downloads, and initiates document processing
    """
    if request.method == 'POST':
        # Get input method (upload or url)
        input_method = request.form.get('input_method', 'upload')
        
        # Get any extra user terms (optional) as a list
        extra = request.form.get('query_terms', '').split(',')
        if extra != ['']:
            query_terms = DEFAULT_QUERY + extra
        else:
            query_terms = DEFAULT_QUERY

        file_path = None
        filename = None
        
        try:
            '''Queue
            # Check if queue is full
            if processing_queue.full():
                return render_template('index.html', 
                                     error="Server is busy processing other documents. Please try again in a few minutes.")
            '''
            # Generate unique task ID
            task_id = str(uuid.uuid4())

            # Create directories
            by_products_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'by-products')
            os.makedirs(by_products_folder, exist_ok=True) # Ensure the subfolder exists
            
            if input_method == 'upload':
                # Handle file upload
                file = request.files.get('pdf_file')
                if not file or file.filename == '':
                    return render_template('index.html', error="Please select a PDF file to upload.")
                
                if not file.filename.lower().endswith('.pdf'):
                    return render_template('index.html', error="Please upload a valid PDF file.")
                
                filename = secure_filename(file.filename)
                file_path = os.path.join(by_products_folder, filename)
                file.save(file_path)

                # Log the file upload
                logging.info(f"File uploaded: {file.filename}")
                
            elif input_method == 'url':
                # Handle URL download
                pdf_url = request.form.get('pdf_url', '').strip()
                if not pdf_url:
                    return render_template('index.html', error="Please enter a PDF URL.")
                
                try:
                    file_path, filename = download_pdf_from_url(pdf_url, by_products_folder)
                    if not file_path:
                        return render_template('index.html', error="Failed to download PDF from the provided URL.")
                except ValueError as e:
                    return render_template('index.html', error=f"Download error: {str(e)}")
                except Exception as e:
                    logging.error(f"Unexpected error downloading from URL: {e}")
                    return render_template('index.html', error="An unexpected error occurred while downloading the PDF.")
            
            else:
                return render_template('index.html', error="Please select an input method.")
            
            # Verify we have a valid file
            if not file_path or not os.path.exists(file_path):
                return render_template('index.html', error="Failed to process the PDF file.")
            
            # Detect document length and calculate time estimate HERE
            document_pages = "unknown"
            estimated_minutes = "unknown"
            time_message = ""

            try:
                doc = fitz.open(file_path)
                document_pages = len(doc)
                doc.close()

                # Calculate estimated time using calculated regression: time = 0.123 * pages (rounded to full minutes)
                estimated_minutes = max(1, round(0.123 * document_pages))  # At least 1 minute
                
                # Create user-friendly message
                if estimated_minutes <= 5:
                    time_message = f"Should complete in about {estimated_minutes} minute{'s' if estimated_minutes > 1 else ''}"
                    encouragement = ""
                elif estimated_minutes <= 15:
                    time_message = f"Estimated processing time: {estimated_minutes} minutes"
                    encouragement = "☕ Perfect time for a quick coffee break!"
                elif estimated_minutes <= 30:
                    time_message = f"Estimated processing time: {estimated_minutes} minutes"
                    encouragement = "📚 Great time to catch up on some reading!"
                elif estimated_minutes <= 60:
                    time_message = f"Estimated processing time: {estimated_minutes} minutes"
                    encouragement = "🍽️ Perfect timing for lunch!"
                else:
                    time_message = f"Estimated processing time: {estimated_minutes} minutes"
                    encouragement = "🎬 That's a long document. Time to watch a movie while we work!"

                logging.info(f"Document analysis: {filename} has {document_pages} pages, estimated time: {estimated_minutes} minutes")
                
            except Exception as page_detection_error:
                logging.warning(f"Could not detect document length for {filename}: {page_detection_error}")
                time_message = "Processing time will vary based on document size"
                encouragement = ""

            processing_thread = threading.Thread(
                target=process_document,
                args=(task_id, file_path, query_terms, filename, document_pages, estimated_minutes),
                daemon=False
            )
            processing_thread.start()

            '''Queue
            # Initialize task tracking
            processing_tasks[task_id] = {
                'status': 'queued',
                'progress': 0,
                'start_time': time.time(),
                'filename': filename,
                #'queue_position': processing_queue.qsize() + 1 #Queue
            }

            # Add task to queue
            processing_queue.put({
                "task_id": task_id,
                "file_path": file_path,
                "query_terms": query_terms,
                "filename": filename,
                "document_pages": document_pages,
                "estimated_minutes": estimated_minutes
            })
            
            # Calculate queue position and estimated wait time
            queue_position = processing_queue.qsize()
            
            # Rough estimate of wait time based on queue position
            if queue_position == 1:
                queue_message = "Your document is next in line!"
                total_wait_minutes = estimated_minutes
            else:
                # Assume average 20 minutes per document ahead
                queue_wait_minutes = (queue_position - 1) * 20
                total_wait_minutes = queue_wait_minutes + estimated_minutes
                queue_message = f"You are #{queue_position} in the queue. Estimated wait: {queue_wait_minutes} minutes before processing starts."
            '''
            
            return render_template('progress.html', 
                                 task_id=task_id,
                                 #queue_position=queue_position,    #Queue
                                 #queue_message=queue_message,      #Queue
                                 original_filename=filename,
                                 document_pages=document_pages,
                                 estimated_minutes=estimated_minutes,
                                 #total_wait_minutes=total_wait_minutes,    #Queue
                                 time_message=time_message,
                                 encouragement=encouragement)
            
        except Exception as e:
            logging.error(f"Error in index route: {e}")
            # Clean up uploaded file if there was an error
            if 'file_path' in locals() and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            return render_template('index.html', error=f"An error occurred: {str(e)}")
    
    # GET request - show the upload form
    return render_template('index.html')

@app.route('/results/<task_id>')
def results(task_id):
    """
    Display results when processing is complete
    """
    task = processing_tasks.get(task_id)

    if task and task.get("result") is not None:
        citations = task.get("result", [])
        original_filename = task.get("original_filename", "analysis")
        
        app.logger.info(f"[RESULTS {task_id}] Processing {len(citations)} citation sources")

        # Process citations to ensure proper quote structure for template
        processed_citations = {}
        total_verified = 0
        total_unverified = 0
        for key, citation in citations.items():
            processed_citation = citation.copy()

            all_quotes = []
            
            # Ensure quotes have proper 3-element structure (index, text, verified)
            if 'quotes' in processed_citation:
                for quote_tuple in processed_citation['quotes']:
                    if len(quote_tuple) >= 3:
                        # Already has verification status
                        all_quotes.append(quote_tuple)
                        if quote_tuple[2]:
                            total_verified += 1
                        else:
                            total_unverified += 1
                    else:
                        # Missing verification status, assume verified for backward compatibility
                        all_quotes.append((quote_tuple[0], quote_tuple[1], True))
                        total_verified += 1

            # Process unverified_quotes if they exist separately
            if 'unverified_quotes' in processed_citation:
                app.logger.info(f"[RESULTS {task_id}] Found {len(processed_citation['unverified_quotes'])} separate unverified quotes for source {key}")
                for quote_tuple in processed_citation['unverified_quotes']:
                    if len(quote_tuple) >= 3:
                        all_quotes.append(quote_tuple)
                        if quote_tuple[2]:
                            total_verified += 1
                        else:
                            total_unverified += 1
                    else:
                        # Assume unverified
                        all_quotes.append((quote_tuple[0], quote_tuple[1], False))
                        total_unverified += 1
            
            # Set the combined quotes
            processed_citation['quotes'] = all_quotes
            # Remove the separate unverified_quotes
            processed_citation.pop('unverified_quotes', None)
            
            processed_citations[key] = processed_citation

        app.logger.info(f"[RESULTS {task_id}] Final quote counts: {total_verified} verified, {total_unverified} unverified")
        
        # Get runtime and token stats
        total_runtime_seconds = task.get("total_runtime_seconds", 0)
        total_tokens_used = task.get("total_tokens_used", 0)
        
        # Get classification data if available
        #classification_data = {} # OLD - replaced with list
        classified_quotes = []
        if "classified_quotes" in task.get("partial_results", {}):
            classified_quotes = task["partial_results"]["classified_quotes"]
            # Create lookup dictionary for quick access
            #for item in classified_quotes:
            #    classification_data[item.get('quote', '')] = item
        
        # ✅ CRITICAL: Sanitize before passing to template
        #classification_data = _sanitize_for_json(classification_data) #OLD
        classified_quotes = _sanitize_for_json(classified_quotes)

        # Get summary data
        summary_data = task.get("summary", {})

        # Format runtime for display
        if total_runtime_seconds > 3600:  # More than 1 hour
            runtime_display = f"{total_runtime_seconds/3600:.1f} hours"
        elif total_runtime_seconds > 60:  # More than 1 minute
            runtime_display = f"{total_runtime_seconds/60:.1f} minutes"
        else:
            runtime_display = f"{total_runtime_seconds:.1f} seconds"

        # Get document pages from task
        document_pages = task.get("document_pages", "N/A")

        return render_template(
            'results.html',
            citations=processed_citations, # NEW: Use processed citations
            classification_data=classified_quotes, # Now sanitized
            summary_data=summary_data, # NEW
            results_folder='results',
            task_id=task_id,
            original_filename=original_filename,
            document_pages=document_pages,
            total_runtime=runtime_display,           
            total_tokens=total_tokens_used,          
            runtime_seconds=total_runtime_seconds    
        )
    else:
        # If task doesn't exist or isn't complete, redirect to progress
        return render_template('progress.html', 
                             task_id=task_id,
                             original_filename=task.get("original_filename", "Unknown") if task else "Unknown",
                             document_pages=task.get("document_pages", "unknown") if task else "Unknown",
                             estimated_minutes=task.get("estimated_time_minutes", "unknown") if task else "Unknown",
                             time_message="",
                             encouragement="")

@app.route('/api/heartbeat')
def heartbeat():
    """Lightweight heartbeat endpoint for Azure health checks"""
    return jsonify({
        "status": "alive",
        "timestamp": time.time(),
        "active_tasks": len(processing_tasks),
        "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024
    })

@app.route('/api/status/<task_id>')
def task_status(task_id):
    """Enhanced status with queue position"""
    task = processing_tasks.get(task_id, {})
    
    # Calculate queue position if still queued
    if task.get('status') == 'queued':
        queue_position = 1
        # Count tasks ahead in queue (simplified)
        for queued_task_id, queued_task in processing_tasks.items():
            if (queued_task.get('status') == 'queued' and 
                queued_task.get('start_time', 0) < task.get('start_time', 0)):
                queue_position += 1
        task['queue_position'] = queue_position
    
    return jsonify(task)

@app.route('/api/detailed_status/<task_id>')
def detailed_task_status(task_id):
    """Detailed status for specific task"""
    if task_id in processing_tasks:
        task = processing_tasks[task_id]
        return jsonify({
            "task_id": task_id,
            "progress": task.get("progress", 0),
            "status": task.get("status", "Unknown"),
            "error": task.get("error"),
            "has_partial_results": task.get("has_partial_results", False),
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,
            "last_update": time.time()
        })
    return jsonify({"error": "Task not found"}), 404

@app.route('/api/progress/<task_id>')
def progress(task_id):
    """
    API endpoint to retrieve task progress with enhanced error handling
    """
    try:
        app.logger.info(f"[API] Progress check for task {task_id}")
        
        if task_id not in processing_tasks:
            app.logger.warning(f"[API] Task {task_id} not found in processing_tasks")
            return jsonify({"error": "Task not found", "task_id": task_id}), 404
        
        task = processing_tasks[task_id]
        app.logger.info(f"[API] Task {task_id} progress: {task.get('progress', 0)}%")
        
        # Check if task has failed FIRST
        if task.get("error") is not None:
            # DEBUG: Log what we're about to return
            partial_results = task.get("partial_results", {})
            has_partial_results = bool(partial_results)
            
            app.logger.error(f"[API] Task {task_id} failed with error: {task['error']}")
            app.logger.error(f"[DEBUG] partial_results: {partial_results}")
            app.logger.error(f"[DEBUG] has_partial_results: {has_partial_results}")
            
            response_data = {
                "progress": task.get("progress", 0),
                "status": task.get("status", "Failed"),
                "failed": True,
                "error": task["error"],
                "traceback": task.get("traceback", "No traceback available"),
                "has_partial_results": has_partial_results,
                "partial_results_summary": partial_results,
                "document_pages": task.get("document_pages", "unknown"),
                "estimated_time_minutes": task.get("estimated_time_minutes", "unknown"),
                "timestamp": time.time(),
                "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024
            }
            
            app.logger.error(f"[DEBUG] Response JSON: {json.dumps(response_data, indent=2)}")
            return jsonify(response_data)
        
        
        # Check if processing is complete
        if task.get("progress", 0) >= 100 and task.get("error") is None:
            app.logger.info(f"[API] Task {task_id} completed successfully")
            return jsonify({
                "progress": 100,
                "status": "Analysis complete",
                "completed": True,
                "redirect": url_for('results', task_id=task_id),
                "document_pages": task.get("document_pages", "unknown"),
                "estimated_time_minutes": task.get("estimated_time_minutes", "unknown"),
                "timestamp": time.time(),
                "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024
            })
        
        # Return normal progress update
        response = {
            "progress": task.get("progress", 0),
            "status": task.get("status", f"Processing... {task.get('progress', 0)}%"),
            "document_pages": task.get("document_pages", "unknown"),
            "estimated_time_minutes": task.get("estimated_time_minutes", "unknown"),
            "timestamp": time.time(),
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,
            "last_heartbeat": task.get("last_heartbeat", time.time())
        }
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"[API] Error in progress endpoint for task {task_id}: {e}")
        app.logger.error(f"[API] Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Progress check failed: {str(e)}"}), 500


@app.route('/download')
def download_results():
    """
    Route to download the entire results folder as a zip file
    with a filename based on the original document
    """
    try:
        # Get the task_id and original filename from query parameters
        task_id = request.args.get('task_id', '')
        original_filename = request.args.get('filename', '')
        
        # Validate task_id
        if not task_id:
            return "Task ID required", 400
        
        # If no filename provided, try to get it from the task
        if not original_filename and task_id and task_id in processing_tasks:
            original_filename = processing_tasks[task_id].get('original_filename', '')
        
        # Create a task-specific zip file
        memory_file = create_results_zip(task_id)
        
        # Create proper download filename
        if original_filename:
            # Remove file extension and clean filename
            base_filename = os.path.splitext(original_filename)[0]
            # Remove any existing '_analysis' suffix to avoid duplication
            download_name = f'{base_filename}_analysis.zip'
        else:
            # Fallback filename with timestamp
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            download_name = f'transport_policy_analysis_{timestamp}.zip'
        
        logging.info(f"Download prepared for task {task_id}: {download_name}")

        return send_file(
            memory_file,
            download_name=download_name,
            as_attachment=True,
            mimetype='application/zip'
        )
    except FileNotFoundError as e:
        logging.error(f"Download error: {e}")
        return f"Error: {e}", 404
    except Exception as e:
        logging.error(f"Unexpected error during download: {e}")
        return "Internal Server Error", 500

@app.route('/api/heartbeat/<task_id>', methods=['POST'])
def task_heartbeat(task_id):
    """Update task heartbeat timestamp"""
    if task_id in processing_tasks:
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        return jsonify({"status": "updated"})
    return jsonify({"status": "not_found"}), 404

@app.route('/api/partial-results/<task_id>')
def get_partial_results(task_id):
    """Get partial results for progressive display"""
    try:
        if task_id not in processing_tasks:
            return jsonify({"error": "Task not found"}), 404
        
        task = processing_tasks[task_id]
        partial_results = task.get("partial_results", {})
        
        # ✅ CRITICAL FIX: Sanitize before returning JSON
        safe_partial = _sanitize_for_json(partial_results)
        
        return jsonify({
            "task_id": task_id,
            "progress": task.get("progress", 0),
            "status": task.get("status", "Processing..."),
            "text_partitions": safe_partial.get("text_partitions", []),
            "total_partitions": safe_partial.get("total_partitions", 0),
            "text_extraction_complete": safe_partial.get("text_extraction_complete", False),
            "search_results": safe_partial.get("search_results", []),
            "total_search_results": safe_partial.get("total_search_results", 0),
            "semantic_search_complete": safe_partial.get("semantic_search_complete", False),
            "extracted_quotes": safe_partial.get("extracted_quotes", {}),
            "quote_extraction_complete": safe_partial.get("quote_extraction_complete", False),
            "classified_quotes": safe_partial.get("classified_quotes", []),
            "classification_complete": safe_partial.get("classification_complete", False),
            "original_filename": task.get("original_filename", "Unknown")
        })
    except Exception as e:
        app.logger.error(f"[PARTIAL-RESULTS] Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/partitions/<task_id>')
def get_document_partitions(task_id):
    """Get document partitions for display"""
    if task_id not in processing_tasks:
        return jsonify({"error": "Task not found"}), 404
    
    task = processing_tasks[task_id]
    partitions = task.get("partial_results", {}).get("text_partitions", [])
    
    return jsonify({
        "partitions": partitions,
        "total_count": task.get("partial_results", {}).get("total_partitions", 0),
        "showing_count": len(partitions)
    })

@app.route('/download-partial')
def download_partial_results():
    """
    Route to download partial results when processing fails
    """
    try:
        # Get the task_id and original filename from query parameters
        task_id = request.args.get('task_id', '')
        original_filename = request.args.get('filename', '')
        
        # Validate task_id
        if not task_id:
            return "Task ID required", 400
        
        # If no filename provided, try to get it from the task
        if not original_filename and task_id and task_id in processing_tasks:
            original_filename = processing_tasks[task_id].get('original_filename', '')
        
        # Verify the task exists and has an error
        if task_id not in processing_tasks:
            return "Task not found", 404
            
        task = processing_tasks[task_id]
        if task.get("error") is None:
            return "No error occurred for this task. Use the regular download instead.", 400
        
        # Create partial results zip file for specific task
        memory_file = create_partial_results_zip(task_id)
        
        # Create proper download filename
        if original_filename:
            base_filename = os.path.splitext(original_filename)[0]
            download_name = f'{base_filename}_partial_results.zip'
        else:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            download_name = f'transport_policy_partial_{timestamp}.zip'
        
        logging.info(f"Partial download prepared for task {task_id}: {download_name}")
        
        return send_file(
            memory_file,
            download_name=download_name,
            as_attachment=True,
            mimetype='application/zip'
        )
    except FileNotFoundError as e:
        logging.error(f"Partial download error: {e}")
        return f"Error: {e}", 404
    except Exception as e:
        logging.error(f"Unexpected error during partial download: {e}")
        return "Internal Server Error", 500

@app.route('/api/status')
def app_status():
    """
    Quick status check to verify app is responsive during processing
    """
    return jsonify({
        "status": "alive",
        "timestamp": time.time(),
        "active_tasks": len(processing_tasks),
        "memory_info": "available"  # You can add memory monitoring here
    })

@app.route('/debug/logs')
def debug_logs():
    """Debug endpoint to see recent errors and API usage"""
    try:
        # Get recent processing task errors
        errors = []
        api_stats = {"total_tasks": len(processing_tasks), "rate_limit_errors": 0, "api_errors": 0}
        
        for task_id, task in processing_tasks.items():
            if task.get('error'):
                error_msg = task['error'].lower()
                if "rate limit" in error_msg:
                    api_stats["rate_limit_errors"] += 1
                if "api" in error_msg or "openai" in error_msg:
                    api_stats["api_errors"] += 1
                    
                errors.append({
                    'task_id': task_id,
                    'error': task['error'],
                    'traceback': task.get('traceback', 'No traceback')[:500],  # Limit length
                    'timestamp': task.get('last_heartbeat', 'Unknown')
                })
        
        return jsonify({
            'recent_errors': errors[-5:],  # Last 5 errors
            'api_statistics': api_stats,
            'active_tasks': len(processing_tasks),
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
        })
    except Exception as e:
        return jsonify({'debug_error': str(e)})
    
@app.route('/debug/task/<task_id>')
def debug_task(task_id):
    """Debug endpoint to see raw task data"""
    if task_id in processing_tasks:
        task = processing_tasks[task_id]
        return jsonify({
            'task_exists': True,
            'task_data': {
                'progress': task.get('progress', 'NOT_SET'),
                'status': task.get('status', 'NOT_SET'),
                'partial_results': task.get('partial_results', 'NOT_SET'),
                'error': task.get('error', 'NOT_SET'),
                'error_type': type(task.get('error')).__name__,
                'has_error': task.get('error') is not None,
                'failed_flag': task.get('failed', 'NOT_SET'),
                'traceback_exists': task.get('traceback') is not None,
                'last_heartbeat': task.get('last_heartbeat', 'NOT_SET'),
                'all_keys': list(task.keys())
            }
        })
    else:
        return jsonify({'task_exists': False, 'task_id': task_id})

@app.route('/api/debug/logs-file')
def debug_logs_file():
    """Read the last 50 lines of the log file directly"""
    try:
        log_file = os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'logs', 'app.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                return jsonify({
                    "last_50_lines": lines[-50:],  # Last 50 lines
                    "total_lines": len(lines),
                    "file_size_kb": os.path.getsize(log_file) / 1024
                })
        else:
            return jsonify({"error": "Log file not found"})
    except Exception as e:
        return jsonify({"error": str(e)})

def background_cleanup():
    """Background thread to clean up old task results"""
    while True:
        try:
            cleanup_old_task_results(age_hours=24)  # Clean up after 24 hours
            time.sleep(3600)  # Run every hour
        except Exception as e:
            logging.error(f"Background cleanup error: {e}")
            time.sleep(3600)  # Still wait an hour before retrying

# Start cleanup thread
logging.info("Starting background cleanup thread...")
cleanup_thread = threading.Thread(target=background_cleanup, daemon=True, name="CleanupThread")
cleanup_thread.start()
logging.info("Background cleanup thread started successfully")

if __name__ == '__main__':

    # For Render.com, use the PORT environment variable
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=True for development
