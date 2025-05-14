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

from flask import Flask, request, render_template, send_file, redirect, url_for
import fitz  # PyMuPDF
import os
import tempfile
import json
import pandas as pd
from werkzeug.utils import secure_filename
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import HumanMessage
from langchain_community.vectorstores import FAISS # https://python.langchain.com/docs/integrations/vectorstores/faiss/
from langchain.docstore.document import Document
import openai
from langchain_community.callbacks import get_openai_callback
import json
from unstructured.partition.pdf import partition_pdf
import shutil
from zipfile import ZipFile
import io
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from quotation_utils import get_quotes
from classification_utils import tagging_classifier_quotes
from general_utils import create_highlighted_pdf

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir() # Temporary folder for file uploads

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
embedding_model = AzureOpenAIEmbeddings(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    deployment = 'embeds'
    )

# Default semantic search query (token-based)
DEFAULT_QUERY = ["transport  vehicles  emissions  BA  electric U  GH  reduction  public G  compared  scenario  transportation  reduce  CO  levels  net es  zero  fuel  vehicle  em  passenger  road  mobility  veh  target  new idad CO  Mt  car ision  transporte  rail  cars  fleet  buses  fuels  traffic  efficiency ículos ar ct e  gas  greenhouse  redu  freight  d  l  share  km o  bio  achieve os  elé els  hydrogen  urban  infrastructure  electr The  hybrid  relative  charging  neutrality eq  é ici )  least  total ado  emission  vé  standards én  aims  e  ambition ’  modes il  carbon  shift as  neutral fu  bus  EV  ré  mov  condition hic  sales  million cción  inter  año  modal  maritime  system  diesel  público  kt  network ules  alternative  cities  percent  heavy re  conditional  Transport  improvement -em  Electric RT  level  use nel  transit  roads  in  light ibles  energ  year rica  goal  aviation  per missions  long  powered  European  consumption arbon ric  lanes  vo  part  walking  sharing  rapport ación  t  bicycle  motor  stations  infra  s duction ov a To  sc  railways  cent  private ías  reductions ). ual r  achieved ada -m condition  élect  ef"]

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

def extract_text_with_unstructured(pdf_path):
    """
    Extracts text and metadata from a PDF file using the `partition_pdf` function.

    Args:
        pdf_path (str): The file path to the PDF document to be processed.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - "text" (str): The extracted text content from a PDF element.
            - "metadata" (dict): Metadata associated with the PDF element, if available.
    """
    # see: https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/partition/pdf.py
    elements = partition_pdf(filename=pdf_path,
                            #strategy='hi_res', # requires Poppler installation
                            coordinates=True,  # return bounding box coordinates for each element extracted via OCR
                            #infer_table_structure=True,  # infer table structure
                            #extract_images_in_pdf=True,  # extract images in PDF
                            languages=['eng','jpn'],  # language detection
                            #extract_image_block_types=["Image", "Table"],  # The types of elements to extract, for use in extracting image blocks as base64 encoded data stored in metadata fields
                            )

    content = []
    for el in elements:
        if el.text:
            content.append({
                "text": el.text.strip(),
                "metadata": el.metadata.to_dict() if el.metadata else {}
            })
    
    # Save content as JSON in the by-products folder
    by_products_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'by-products')
    os.makedirs(by_products_folder, exist_ok=True)  # Ensure the subfolder exists
    json_path = os.path.join(by_products_folder, os.path.join(by_products_folder, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_partition.json"))
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(content, json_file, ensure_ascii=False, indent=4)
    
    return content

def semantic_search_FAISS(text_chunks, queries, filename, sim_threshold=0.69, k=20): 
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
    
    # Create and save the vector store
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(vector_store_path)

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

def semantic_search_Chroma(text_chunks, queries, filename, sim_threshold=0.65, k=20):
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
        
        print(f"Total tokens used: {cb.total_tokens}")

    return quotes_dict

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
    memory_file = io.BytesIO()
    
    with ZipFile(memory_file, 'w') as zf:
        for root, dirs, files in os.walk(results_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate the relative path for the file in the zip
                relative_path = os.path.relpath(file_path, app.config['UPLOAD_FOLDER'])
                zf.write(file_path, relative_path)
    
    # Move to the beginning of the BytesIO buffer
    memory_file.seek(0)
    return memory_file

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    TODO: add highlighting, classification
    """
    if request.method == 'POST':
        file = request.files['pdf_file']
        # Get any extra user terms (optional) as a list
        extra = request.form['query_terms'].split(',')
        if extra != ['']:
            query_terms = DEFAULT_QUERY + extra
        else:
            query_terms = DEFAULT_QUERY

        if file:
            filename = secure_filename(file.filename)
            by_products_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'by-products')
            os.makedirs(by_products_folder, exist_ok=True)  # Ensure the subfolder exists
            file_path = os.path.join(by_products_folder, filename)
            file.save(file_path)

            #citations, highlighted_path = extract_text_and_highlight(file_path, query_terms)
            citations = extract_text_and_highlight(file_path, query_terms)

            classified = classify_quotes(citations, file_path)

            postprocess_results(file_path, classified, filename)

            return render_template(
                'results.html',
                #citations=classified,
                citations=citations,
                #highlighted_pdf=os.path.basename(highlighted_path),
                results_folder='results'
            )
        
    return render_template('index.html')

@app.route('/download/<filename>')
#deprecated: using download_results instead
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

@app.route('/download_results')
def download_results():
    """
    Route to download the entire results folder as a zip file
    """
    memory_file = create_results_zip()
    return send_file(
        memory_file,
        download_name='analysis_results.zip',
        as_attachment=True,
        mimetype='application/zip'
    )

if __name__ == '__main__':
    app.run(debug=True, port=5006)  # Set debug=True for development

