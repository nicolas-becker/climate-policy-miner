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

from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify
import fitz  # PyMuPDF
import os
import tempfile
import json
import pandas as pd
import uuid
import threading
import logging
import time
import random
import requests
import mimetypes
from urllib.parse import urlparse
from werkzeug.utils import secure_filename
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import HumanMessage
from langchain_community.vectorstores import FAISS # https://python.langchain.com/docs/integrations/vectorstores/faiss/
from langchain.docstore.document import Document
from langchain_community.callbacks import get_openai_callback
from openai import InternalServerError
import json
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
import psutil

try:
    from .quotation_utils import get_quotes
    from .classification_utils import tagging_classifier_quotes
    from .general_utils import create_highlighted_pdf
except ImportError:
    # Fallback for direct execution
    from quotation_utils import get_quotes
    from classification_utils import tagging_classifier_quotes
    from general_utils import create_highlighted_pdf

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir() # Temporary folder for file uploads

# Set up logging
log_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'logs')
os.makedirs(log_folder, exist_ok=True)  # Ensure the logs folder exists
log_file = os.path.join(log_folder, 'app.log')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Log the start of the application
logging.info("============" \
             "App started." \
             "============")

# Store for tracking processing tasks
processing_tasks = {}

# Environment variable setup for LangChain tracing
# Uncomment the following lines to enable tracing with LangChain (requires valid API key)
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
#os.environ["LANGCHAIN_PROJECT"] = "climate_policy_pipeline"
os.environ["LANGCHAIN_TRACING_V2"] = "false" # deactivate tracing for now, as api key is required to be updated to v2: "error":"Unauthorized: Using outdated v1 api key. Please use v2 api key."

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
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    deployment = 'embeds'
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

# Add these imports at the top with your other imports
import requests
from urllib.parse import urlparse
import mimetypes

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
def extract_text_with_unstructured(pdf_path, min_lenght=10, max_length=10000, apply_cleaning=True):
    """
    Extracts text and metadata from a PDF file using the `partition_pdf` function.
    Now includes general text cleaning using clean_elements().

    Args:
        pdf_path (str): The file path to the PDF document to be processed.
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
        by_products_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'by-products')
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

def semantic_search_FAISS(text_chunks, queries, filename, embedding_model, sim_threshold=0.69, k=20, max_retries=3): 
    """
    Perform a semantic search over a collection of text chunks using a vector store with FAISS.
    The vector store is saved for future use.

    Args:
        text_chunks (list of dict): A list of dictionaries where each dictionary represents a text chunk.
            Each dictionary must have a 'text' key containing the text content, and optionally a 'metadata' key
            containing additional metadata (e.g., page number).
        queries (list of str): A list of query strings to search for in the text chunks.
        filename (str): The name of the file being processed (used for naming the vector store)
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
    vector_store_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'vector_stores')
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

def semantic_search_Chroma(text_chunks, queries, filename, embedding_model, sim_threshold=0.65, k=20):
    """
    Perform a semantic search over a collection of text chunks using Chroma with cosine similarity.
    The vector store is saved for future use.

    Args:
        text_chunks (list of dict): A list of dictionaries where each dictionary represents a text chunk.
            Each dictionary must have a 'text' key containing the text content, and optionally a 'metadata' key
            containing additional metadata (e.g., page number).
        queries (list of str): A list of query strings to search for in the text chunks.
        filename (str): The name of the file being processed (used for naming the vector store)
        sim_threshold (float, optional): The minimum cosine similarity score to consider a match. Defaults to 0.65.
        k (int, optional): The number of top matches to retrieve for each query. Defaults to 20.

    Returns:
        list of dict: A list of dictionaries representing the matched results. Each dictionary contains:
            - "text" (str): The content of the matched text chunk.
            - "page" (int or str): The page number from the metadata, or "N/A" if not provided.
            - "score" (float): The cosine similarity score of the match (higher is better, threshold of sim_threshold is applied).
    """
    
    documents = []
    for chunk in text_chunks:
        metadata = chunk.get("metadata", {})
        if isinstance(metadata, dict):
            # Only filter if metadata is a dictionary
            try:
                metadata = filter_complex_metadata(metadata)
            except Exception as e:
                # Fallback if filtering fails
                metadata = {k: str(v) for k, v in metadata.items() if k != "metadata"}
        else:
            # Convert non-dict metadata to a simple dict
            metadata = {"raw_metadata": str(metadata)}
        
        documents.append(Document(page_content=chunk['text'], metadata=metadata))
    
    # Create the vector store directory if it doesn't exist
    vector_store_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'vector_stores')
    os.makedirs(vector_store_dir, exist_ok=True)
    
    # Create a unique name for this document's vector store
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    vector_store_path = os.path.join(vector_store_dir, f"{base_filename}_chroma")
    
    # Create and save the vector store with cosine similarity
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=base_filename,
        persist_directory=vector_store_path
    )
    vectorstore.persist()  # Save the vectorstore to disk

    # Perform similarity search for each query
    matches = []
    for query in queries:
        # Getting results with scores (Chroma uses cosine similarity by default)
        results = vectorstore.similarity_search_with_score(query, k=len(text_chunks))
        
        for res, score in results:
            # In Chroma, higher score = more similar with cosine similarity
            if score >= sim_threshold:  # Apply the similarity threshold
                matches.append({
                    "text": res.page_content,
                    "filename": res.metadata.get("filename", "N/A"),
                    "page": res.metadata.get("page_number", "N/A"),
                    "score": score
                })
        
    return matches

def extract_quotes(matches, filename):
    """
    Process semantic search matches to extract quotes using get_quotes function.
    
    Args:
        matches (list): List of matches from semantic search
        filename (str): Base name of the file being processed
    
    Returns:
        list: Original matches with extracted quotes added
        int: Total tokens used during the extraction process
    """
    # Convert matches into the dictionary format needed for get_quotes
    doc_dict = {}
    for i, match in enumerate(matches):
        doc_dict[str(i)] = {
            'filename' : match["filename"],
            'content': match["text"],
            'type': 'text',  # Assuming all are text entries
            'page_number': match["page"],
            'score': match["score"]
        }
    
    # Create the by-products folder if it doesn't exist
    by_products_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'by-products')
    os.makedirs(by_products_folder, exist_ok=True)
    
    # Extract quotes using the provided utility
    with get_openai_callback() as cb:
        quotes_dict = get_quotes(LLM, doc_dict)
        
        # Save the quotes to a JSON file
        quotes_path = os.path.join(by_products_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_extracted-quotes.json")
        with open(quotes_path, 'w', encoding='utf-8') as f:
            json.dump(quotes_dict, f, ensure_ascii=False, indent=4)
        
        # Log the total tokens used
        total_tokens = cb.total_tokens
        logging.info(f"Total tokens used for quote extraction: {total_tokens}")

    return quotes_dict, total_tokens

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

def extract_text_and_highlight(pdf_path, query_terms):
    """
    TODO: Highlighting missing
    """
    unstructured_text = extract_text_with_unstructured(pdf_path)
    chunks = semantic_search_FAISS(unstructured_text, query_terms, os.path.splitext(os.path.basename(pdf_path))[0])
    citations = extract_quotes(chunks, os.path.basename(pdf_path))
    #highlighted_path = highlight_terms_in_pdf(pdf_path, query_terms)
    return citations#, highlighted_path

def classify_quotes(quotes_dict, file_directory):
    """
    Step 4 of the Pipeline:
    -----------------------
    This function classifies the extracted quotes into predefined categories.
    
    Params:
        quotes_dict (dict): A dictionary containing the previously extracted quotes.
        file_directory (str): The path to the directory containing the input data and the result folder.
    Returns:
        pd.DataFrame : A pandas DF containing the classified quotes.
    """
    # Classification of quotes into predefined categories
    output_df = tagging_classifier_quotes(quotes_dict=quotes_dict, llm=LLM, fewshot=True)

    # Save results in Excel and CSV{LLM.model_name}_{namespace}
    by_products_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'by-products')
    os.makedirs(by_products_folder, exist_ok=True)  # Ensure the subfolder exists
    output_df.to_excel(os.path.join(by_products_folder, f"{os.path.splitext(os.path.basename(file_directory))[0]}_fewshot-tagging_{LLM.model_name}.xlsx"))
    output_df.to_csv(os.path.join(by_products_folder, f"{os.path.splitext(os.path.basename(file_directory))[0]}_fewshot-tagging_{LLM.model_name}.csv"))

    # TODO: Save the classified quotes to a JSON file
#    classified_quotes_path = os.path.join(by_products_folder, f"{os.path.splitext(os.path.basename(file_directory))[0]}_classified-quotes.json")
#    with open(classified_quotes_path, 'w', encoding='utf-8') as f:
#        json.dump(classified_quotes, f, ensure_ascii=False, indent=4)


    return output_df

def postprocess_results(file_directory, output_df, filename):
    """
    Post-processing of Results:
    -----------------------
    This function performs several postprocessing steps on the output DataFrame from the pipeline.
    It generates highlighted PDFs, and saves specific subsets of the data (targets, mitigation measures, 
    and adaptation measures) to both Excel and CSV files.

    Params:
        input_file (str): The directory of the input file.
        output_df (pd.DataFrame): The DataFrame containing the results of the pipeline.
        filename (str): The name of the file.
    Returns:
        None
    """

    # create output folder
    output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'output')
    os.makedirs(output_folder, exist_ok=True)  # Ensure the subfolder exists

    # highlight quotes
    create_highlighted_pdf(file_directory, 
                            quotes=output_df['quote'],
                            output_path=os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_highlighted.pdf"),
                            df = output_df
                        )

    #  targets
    targets_output_df = output_df[output_df['target']=='True']
    targets_output_df=targets_output_df[['quote', 'page', 'target_labels']]
    
    #  mitigation
    mitigation_output_df = output_df[output_df['mitigation_measure']=='True']
    mitigation_output_df = mitigation_output_df[['quote', 'page', 'measure_labels']]
    
    #  adaptation
    adaptation_output_df = output_df[output_df['adaptation_measure']=='True']
    adaptation_output_df = adaptation_output_df[['quote', 'page', 'measure_labels']]

    #  save to single Excel file
    #  create an Excel writer object
    with pd.ExcelWriter(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(filename))[0]}_results.xlsx")) as writer:
    
        # use to_excel function and specify the sheet_name and index 
        # to store the dataframe in specified sheet
        targets_output_df.to_excel(writer, sheet_name="Targets", index=False)
        mitigation_output_df.to_excel(writer, sheet_name="Mitigation", index=False)
        adaptation_output_df.to_excel(writer, sheet_name="Adaptation", index=False)


# Default provided
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

def create_results_zip():
    """
    Creates a zip file containing the entire results folder
    
    Returns:
        io.BytesIO: An in-memory zip file containing the results folder
    """
    results_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
    
    # Check if the results folder exists and contains files
    if not os.path.exists(results_folder) or not any(os.scandir(results_folder)):
        raise FileNotFoundError("The results folder is empty or does not exist.")
    
    memory_file = io.BytesIO()
    
    with ZipFile(memory_file, 'w') as zf:
        for root, dirs, files in os.walk(results_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate the relative path for the file in the zip
                relative_path = os.path.relpath(file_path, app.config['UPLOAD_FOLDER'])
                zf.write(file_path, relative_path)
     
        # Add the log file to the zip    
        log_file = os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'logs', 'app.log')
        if os.path.exists(log_file):
            zf.write(log_file, os.path.relpath(log_file, app.config['UPLOAD_FOLDER']))

    # Move to the beginning of the BytesIO buffer
    memory_file.seek(0)
    return memory_file

def create_partial_results_zip():
    """
    Creates a zip file containing any available results, even if processing failed
    
    Returns:
        io.BytesIO: An in-memory zip file containing available results
    """
    results_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
    
    # Check if the results folder exists
    if not os.path.exists(results_folder):
        raise FileNotFoundError("No results folder found.")
    
    memory_file = io.BytesIO()
    files_added = 0
    
    with ZipFile(memory_file, 'w') as zf:
        # Add any files that exist in the results folder structure
        for root, dirs, files in os.walk(results_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate the relative path for the file in the zip
                relative_path = os.path.relpath(file_path, app.config['UPLOAD_FOLDER'])
                zf.write(file_path, relative_path)
                files_added += 1
        
        # If no files were found, add a readme explaining the situation
        if files_added == 0:
            readme_content = """PARTIAL RESULTS README
============================

This download contains partial results from a processing task that encountered an error.

Unfortunately, no output files were generated before the error occurred.
However, you may find useful information in the error details provided in the web interface.

For support, please provide:
1. The error message and traceback
2. The original document you were trying to process
3. Any query terms you specified

"""
            zf.writestr("README_PARTIAL_RESULTS.txt", readme_content)

    # Move to the beginning of the BytesIO buffer
    memory_file.seek(0)
    return memory_file

def process_document(task_id, file_path, query_terms, filename):
    """
    Process a document in the background and update progress.
    
    Args:
        task_id (str): Unique ID for this processing task
        file_path (str): Path to the PDF file
        query_terms (list): Terms to search for
        filename (str): Original filename
    """
    try:
        # Initialize progress tracking
        processing_tasks[task_id] = {
            "progress": 0,
            "status": "Starting document analysis...",
            "result": None,
            "error": None,
            "traceback": None,
            "original_filename": filename,  # Store the original filename
            "partial_results": {},  # Store partial results,
            "last_heartbeat": time.time()  # Add heartbeat timestamp
        }
        
        # Start total runtime timer
        total_start_time = time.time()
        total_tokens_used = 0  # Initialize total token counter
        
        # Step 0: Initialize task
        processing_tasks[task_id]["status"] = "Starting text extraction..."
        processing_tasks[task_id]["progress"] = 5
        processing_tasks[task_id]["last_heartbeat"] = time.time()  # Add heartbeat timestamp

        # Simulate progress during long operations
        processing_tasks[task_id]["status"] = "Analyzing document structure..."
        processing_tasks[task_id]["progress"] = 10
        processing_tasks[task_id]["last_heartbeat"] = time.time()

        # Step 1: Extract text with Unstructured (20%)
        start_time = time.time()
        processing_tasks[task_id]["status"] = "Extracting text from document..."
        unstructured_text = extract_text_with_unstructured(file_path)
        processing_tasks[task_id]["progress"] = 20
        processing_tasks[task_id]["status"] = "Text extraction completed."
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        processing_tasks[task_id]["partial_results"]["text_extraction"] = True  # Mark as completed
        preprocessing_time = time.time() - start_time
        logging.info(f"Text extraction completed for file: {filename} in {preprocessing_time:.2f} seconds.")
        
        # Memory cleanup
        gc.collect()

        # Step 2: Perform semantic search (40%)
        start_time = time.time()
        processing_tasks[task_id]["status"] = "Preparing semantic search..."
        processing_tasks[task_id]["progress"] = 25
        processing_tasks[task_id]["last_heartbeat"] = time.time()

        processing_tasks[task_id]["status"] = "Performing semantic search..."
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        chunks = semantic_search_FAISS(unstructured_text, query_terms, 
                                      os.path.splitext(os.path.basename(file_path))[0],
                                      embedding_model=EMBEDDING)
        processing_tasks[task_id]["progress"] = 30
        processing_tasks[task_id]["status"] = "Semantic search completed."
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        processing_tasks[task_id]["partial_results"]["semantic_search"] = True  # Mark as completed
        retrieval_time = time.time() - start_time
        logging.info(f"Semantic search completed for file: {filename} in {retrieval_time:.2f} seconds.")
        
        # Memory cleanup
        gc.collect()

        # Step 3: Extract quotes (60%)
        start_time = time.time()
        processing_tasks[task_id]["status"] = "Extracting quotes from relevant sections..."
        processing_tasks[task_id]["progress"] = 40
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        citations, tokens_used = extract_quotes(chunks, os.path.basename(file_path))
        total_tokens_used += tokens_used  # Update total tokens used
        processing_tasks[task_id]["progress"] = 60
        processing_tasks[task_id]["status"] = "Quote extraction completed."
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        processing_tasks[task_id]["partial_results"]["quote_extraction"] = True  # Mark as completed
        quotation_time = time.time() - start_time
        logging.info(f"Quote extraction completed for file: {filename} in {quotation_time:.2f} seconds. Tokens used: {tokens_used}.")
        
        # Memory cleanup
        gc.collect()

        # Step 4: Classify quotes (80%)
        start_time = time.time()
        processing_tasks[task_id]["status"] = "Preparing AI classification..."
        processing_tasks[task_id]["progress"] = 65
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        processing_tasks[task_id]["status"] = "Classifying extracted quotes..."
        processing_tasks[task_id]["progress"] = 70
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        with get_openai_callback() as cb:
            classified = classify_quotes(citations, file_path)
            tokens_used = cb.total_tokens
            total_tokens_used += tokens_used  # Update total tokens used
        processing_tasks[task_id]["progress"] = 80
        processing_tasks[task_id]["status"] = "AI classification completed."
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        processing_tasks[task_id]["partial_results"]["classification"] = True  # Mark as completed
        classification_time = time.time() - start_time
        logging.info(f"Quote classification completed for file: {filename} in {classification_time:.2f} seconds. Tokens used: {tokens_used}.")

        # Memory cleanup
        gc.collect()

        # Step 5: Post-process results (100%)
        pathe = time.time()
        processing_tasks[task_id]["status"] = "Finalizing results..."
        processing_tasks[task_id]["progress"] = 90
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        postprocess_results(file_path, classified, filename)
        processing_tasks[task_id]["progress"] = 100
        processing_tasks[task_id]["status"] = "Analysis complete"
        processing_tasks[task_id]["last_heartbeat"] = time.time()
        processing_tasks[task_id]["partial_results"]["postprocessing"] = True  # Mark as completed
        postprocessing_time = time.time() - start_time
        logging.info(f"Post-processing completed for task_id: {task_id}, file: {filename} in {postprocessing_time:.2f} seconds.")

        # Calculate total runtime
        total_elapsed_time = time.time() - total_start_time

        logging.info(f"Total processing time for file: {filename} was {total_elapsed_time:.2f} seconds.")
        logging.info(f"Total tokens used for file: {filename} was {total_tokens_used}.")

        # Store results for retrieval
        processing_tasks[task_id]["status"] = "Analysis complete"
        processing_tasks[task_id]["result"] = citations
        
    except Exception as e:
        # Capture the full traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_traceback = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # Handle any errors with detailed information
        processing_tasks[task_id]["error"] = str(e)
        processing_tasks[task_id]["traceback"] = error_traceback
        processing_tasks[task_id]["status"] = f"Error: {str(e)}"
        logging.error(f"Error processing document: {e}")
        logging.error(f"Traceback: {error_traceback}")
        app.logger.error(f"Task {task_id} failed: {e}")

@app.route('/health')
def health_check():
    """Health check endpoint for Render.com"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "port": os.environ.get('PORT', 'not set'),
        "service": "climate-policy-miner"
    }), 200

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
            # Create directories
            by_products_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'by-products')
            os.makedirs(by_products_folder, exist_ok=True) # Ensure the subfolder exists
            
            if input_method == 'upload':
                # Handle file upload
                file = request.files.get('pdf_file')
                if not file or file.filename == '':
                    return render_template('index.html', error="Please select a PDF file to upload.")
                
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
                return render_template('index.html', error="Invalid input method selected.")
            
            # Verify we have a valid file
            if not file_path or not os.path.exists(file_path):
                return render_template('index.html', error="Failed to process the PDF file.")
            
            # Generate a unique task ID
            task_id = str(uuid.uuid4())
            
            # Start processing in a background thread
            thread = threading.Thread(target=process_document, 
                                    args=(task_id, file_path, query_terms, filename))
            thread.daemon = True
            thread.start()
            
            # Redirect to progress page
            return render_template('progress.html', task_id=task_id, original_filename=filename)
            
        except Exception as e:
            logging.error(f"Error in index route: {e}")
            return render_template('index.html', error=f"An error occurred: {str(e)}")
            
    return render_template('index.html')

@app.route('/results/<task_id>')
def results(task_id):
    """
    Display results when processing is complete
    """
    if task_id in processing_tasks and processing_tasks[task_id]["result"] is not None:
        citations = processing_tasks[task_id]["result"]
        original_filename = processing_tasks[task_id].get("original_filename", "analysis")
        return render_template(
            'results.html',
            citations=citations,
            results_folder='results',
            task_id=task_id,
            original_filename=original_filename
        )
    else:
        # If task doesn't exist or isn't complete, redirect to progress
        return render_template('progress.html', task_id=task_id)

@app.route('/api/heartbeat')
def heartbeat():
    """Lightweight heartbeat endpoint for Azure health checks"""
    return jsonify({
        "status": "alive",
        "timestamp": time.time(),
        "active_tasks": len(processing_tasks),
        "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024
    })

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
    API endpoint to retrieve task progress
    """
    if task_id in processing_tasks:
        task = processing_tasks[task_id]
        response = {
            "progress": task.get("progress", 0),
            "status": task.get("status", f"Processing... {task.get('progress', 0)}%"),
            "timestamp": time.time(),  # Add timestamp for debugging
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,  # Add memory info
        }
        
        # If processing is complete, include redirect URL
        if task["progress"] == 100 and task.get("error") is None:
            response["redirect"] = url_for('results', task_id=task_id)
            response["completed"] = True
        
        # If there was an error, include it and the traceback
        if task.get("error") is not None:
            response["error"] = task["error"]
            response["traceback"] = task.get("traceback", "")
            response["failed"] = True
            
            # Check if any partial results are available for download
            partial_results = task.get("partial_results", {})
            has_partial_results = any(partial_results.values())
            response["has_partial_results"] = has_partial_results
            response["partial_results_summary"] = partial_results

        return jsonify(response)
    else:
        return jsonify({"error": "Task not found", "task_id": task_id}), 404


@app.route('/download')
def download_results():
    """
    Route to download the entire results folder as a zip file
    with a filename based on the original document
    """
    try:
        # Get the task_id and original filename from query parameters
        task_id = request.args.get('task_id', '')
        original_filename = request.args.get('filename', 'analysis')
        
        # Create zip file
        memory_file = create_results_zip()
        
        # Use the original filename (without extension) for the download
        base_filename = os.path.splitext(original_filename)[0]
        download_name = f'{base_filename}_analysis.zip'
        
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


@app.route('/download-partial')
def download_partial_results():
    """
    Route to download partial results when processing fails
    """
    try:
        # Get the task_id and original filename from query parameters
        task_id = request.args.get('task_id', '')
        original_filename = request.args.get('filename', 'analysis')
        
        # Verify the task exists and has an error
        if task_id not in processing_tasks:
            return "Task not found", 404
            
        task = processing_tasks[task_id]
        if task.get("error") is None:
            return "No error occurred for this task. Use the regular download instead.", 400
        
        # Create partial results zip file
        memory_file = create_partial_results_zip()
        
        # Use the original filename (without extension) for the download
        base_filename = os.path.splitext(original_filename)[0]
        download_name = f'{base_filename}_partial_results.zip'
        
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

if __name__ == '__main__':
    # For Render.com, use the PORT environment variable
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=True for development
