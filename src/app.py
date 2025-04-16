#%%
import os
import shutil
import tempfile
import pickle
import logging
import json
import time as t
import pandas as pd

from dotenv import load_dotenv
from flask import Flask, request, render_template, send_file, redirect, url_for
from unstructured.staging.base import convert_to_dataframe
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from codecarbon import track_emissions

from preprocessing_utils import set_file, unstructured_api_wo_chunking, categorize_elements, summarize_elements, summarize_text, summarize_table, summarize_image, add_documents_to_vectorstore, chunk_elements
from retrieval_utils import get_docs_from_vectorstore
from quotation_utils import get_quotes
from classification_utils import tagging_classifier_quotes
from general_utils import load_config, setup_logger, normalize_filename, create_highlighted_pdf, namespace_exists

# Load configuration
config_file = 'src/config.json'
config = load_config(config_file)
api_url = config.get('api_url', 'https://default.com/api')
timeout = config.get('timeout', 10)
enable_logging = config.get('enable_logging', False)

# Combination of token_query and keywords
COMBINED_QUERY =  config.get('QUERY', "transport  vehicles  emissions  BA  electric U  GH  reduction  public G  compared  scenario  transportation  reduce  CO  levels  net es  zero  fuel  vehicle  em  passenger  road  mobility  veh  target  new idad CO  Mt  car ision  transporte  rail  cars  fleet  buses  fuels  traffic  efficiency ículos ar ct e  gas  greenhouse  redu  freight  d  l  share  km o  bio  achieve os  elé els  hydrogen  urban  infrastructure  electr The  hybrid  relative  charging  neutrality eq  é ici )  least  total ado  emission  vé  standards én  aims  e  ambition ’  modes il  carbon  shift as  neutral fu  bus  EV  ré  mov  condition hic  sales  million cción  inter  año  modal  maritime  system  diesel  público  kt  network ules  alternative  cities  percent  heavy re  conditional  Transport  improvement -em  Electric RT  level  use nel  transit  roads  in  light ibles  energ  year rica  goal  aviation  per missions  long  powered  European  consumption arbon ric  lanes  vo  part  walking  sharing  rapport ación  t  bicycle  motor  stations  infra  s duction ov a To  sc  railways  cent  private ías  reductions ). ual r  achieved ada -m condition  élect  ef Transport Energy Net Zero Mitigation Adaptation Conditional Target Measure")

# Chunking flag
CHUNK = config.get('CHUNK', True)

# Fewshot flag
FEWSHOT = config.get('FEWSHOT', True)

# Set logging
LOGFILE = config.get('logging', {'file':'logfile.log'}).get('file', 'logfile.log')
logger = setup_logger(__name__, LOGFILE)

# Variables from .env file
load_dotenv()
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_VISION = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_VISION"]
logger.info("Loaded environment variables")
logger.info(f"Deployment name: {AZURE_OPENAI_CHAT_DEPLOYMENT_NAME}")


# Language models
EMBEDDINGS = AzureOpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,
                                   deployment = 'embeds'
                                   )
#  Text model - Azure OpenAI - currently gpt-35-turbo --> https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#gpt-4-and-gpt-4-turbo-models
TEXT_MODEL = AzureChatOpenAI(
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    model_version='2024-07-18',
    temperature=0
)
logger.info(f"Text model: {TEXT_MODEL.model_name}")	

#  Vision model - Azure OpenAI - currently gpt-4o-mini --> https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#gpt-4-and-gpt-4-turbo-models
VISION_MODEL = AzureChatOpenAI(
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_VISION,
    model_version='2024-07-18',
    temperature=0
)
logger.info(f"Text model: {VISION_MODEL.model_name}")	


print('model name:', TEXT_MODEL.model_name, os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"])
#%% 


# Directories for uploads and results
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Create an instance of the Flask class. This instance will be our WSGI application.
app = Flask(__name__)

@app.route('/')
def home():
    """Render the landing page with a list of result zip files."""
    # List all zip files available in the results folder -- TODO: dispaly the list of files in the landing page
    result_files = [
        f for f in os.listdir(RESULT_FOLDER) if f.endswith('.zip') and os.path.isfile(os.path.join(RESULT_FOLDER, f))
    ]
    return render_template('landing.html', result_files=result_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis."""
    if 'file' not in request.files:
        return "No file provided", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Process the file and generate results
    result_folder = process_file(file_path)
    
    # Prepare the results for download
    result_zip = shutil.make_archive(result_folder, 'zip', result_folder)
    download_filename = os.path.basename(result_zip)  # Get the zip file name
    
    return render_template('result.html', download_path=f'/download/{download_filename}')

@app.route('/start_pipeline', methods=['POST'])
def start_pipeline():
    """Render the file upload form and handle file uploads."""

    # Check if a file is submitted
    if 'file' not in request.files:
        return "No file provided", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Process the file and generate results
    result_folder = process_file(file_path)

    # Prepare the results for download
    result_zip = shutil.make_archive(result_folder, 'zip', result_folder)
    download_filename = os.path.basename(result_zip)

    # Redirect to results page
    return render_template('result.html', download_path=f'/download/{download_filename}')


@app.route('/download/<path:filename>')
def download_file(filename):
    """Serve the zip file for download."""
    file_path = os.path.join(os.getcwd(), RESULT_FOLDER, filename)
    
    if not os.path.exists(file_path):
        return f"File not found: {file_path}", 404
    return send_file(file_path, as_attachment=True)


@app.route('/clear', methods=['POST'])
def clear_files():
    """Clear all uploaded and result files."""
    shutil.rmtree(UPLOAD_FOLDER)
    shutil.rmtree(RESULT_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    return redirect(url_for('home'))

@track_emissions
def process_file(uploaded_file_path):
    """
    TODO --> to be migrated to pipeline_module.py 

    Generates a folder with results for download.
    """
    logger.info("\n\n========================= Start Pipeline =========================\n")
    total_time = t.time()
    total_cost = 0
    total_tokens = 0

    try:
        # Step 0: Create a unique result folder for the results and extract filename
        filename, file_directory = step_0_create_folder(uploaded_file_path)
        logger.info(f"Step 0 took {t.time() - total_time} seconds.")

        # Step 1: Preprocess the input file and set up a vector store
        step_time = t.time()
        with get_openai_callback() as cb:    
            elements_chunked, namespace, vectorstore = step_1_preprocess(uploaded_file_path, filename, file_directory)
            logger.info(f"Preprocessing Cost (USD): ${format(cb.total_cost, '.6f')}")
            logger.info(f"Preprocessing Tokens: {cb.total_tokens}")
            total_cost = total_cost + cb.total_cost
            total_tokens = total_tokens + cb.total_tokens
        logger.info(f"Preprocessing took {t.time() - step_time} seconds.")

        # Step 2: Retrieve elements from the vector store
        step_time = t.time()
        with get_openai_callback() as cb:
            retrieved_elements = step_2_retrieve(elements_chunked, namespace, vectorstore, file_directory)
            logger.info(f"Retrieval Cost (USD): ${format(cb.total_cost, '.6f')}")
            logger.info(f"Retrieval Tokens: {cb.total_tokens}")
            total_cost = total_cost + cb.total_cost
            total_tokens = total_tokens + cb.total_tokens
        logger.info(f"Retrieval took {t.time() - step_time} seconds.")

        # Step 3: Extract quotes from the retrieved elements
        step_time = t.time()
        with get_openai_callback() as cb:
            quotes_dict = step_3_extract_quotes(retrieved_elements, file_directory, namespace)
            logger.info(f"Quote Extraction Cost (USD): ${format(cb.total_cost, '.6f')}")
            logger.info(f"Quote Extraction Tokens: {cb.total_tokens}")
            total_cost = total_cost + cb.total_cost
            total_tokens = total_tokens + cb.total_tokens
        logger.info(f"Quotation took {t.time() - step_time} seconds.")

        # Step 4: Classify the extracted quotes
        step_time = t.time()
        with get_openai_callback() as cb:
            output_df = step_4_classify_quotes(quotes_dict, file_directory, namespace)
            logger.info(f"Classification Cost (USD): ${format(cb.total_cost, '.6f')}")
            logger.info(f"Classification Tokens: {cb.total_tokens}")
            total_cost = total_cost + cb.total_cost
            total_tokens = total_tokens + cb.total_tokens
        logger.info(f"Classification took {t.time() - step_time} seconds.")

        # Post-process the results
        step_time = t.time()
        mitigation_output_df, adaptation_output_df = postprocess_results(uploaded_file_path, output_df, file_directory, namespace)
        logger.info(f"Post-processing took {t.time() - step_time} seconds.")

        additional_file = os.path.join(file_directory, 'summary.txt')
        with open(additional_file, 'w') as summary:
            summary.write(f"filename: {filename}\n")
            summary.write(f"file_directory: {file_directory}\n")
            summary.write(f"namespace: {namespace}\n")
            summary.write(f"vectorstore: {vectorstore}\n")
        
    
    except Exception as e:
        logging.exception(f"Pipeline exit with an error: {e}")
        
    finally:
        total_time = t.time() - total_time
        logger.info(f"Pipeline took {total_time} seconds.")
        logger.info(f"Total Cost (USD): ${format(total_cost, '.6f')}")
        logger.info(f"Total Tokens: {total_tokens}")
        logger.info("\n\n========================= End Pipeline =========================\n")
    
    return file_directory

def step_0_create_folder(input_path):
    """
    Step 0 of the Pipeline:
    -----------------------
    Creates a unique result folder structure based on the input file path.
    This function extracts the filename from the given input path, and then creates a directory structure
    for storing results, by-products, figures, and output files. If the directories already exist, they 
    will not be recreated.
    
    Params:
        input_path (str): The path to the input file.
    Returns:
        tuple: A tuple containing the filename (str) and the path to the created directory (str).
    """
    # Extract filename from path
    filename = os.path.basename(input_path).split('.')[0]
    file_directory = os.path.join(RESULT_FOLDER, filename)
    
    # Create a unique result folder
    if not os.path.exists(f'{os.path.join(os.getcwd(), file_directory)}'):
        os.makedirs(file_directory, exist_ok=True)
        os.makedirs(f'{file_directory}/by-products/', exist_ok=True)
        os.makedirs(f'{file_directory}/by-products/figures/', exist_ok=True)
        os.makedirs(f'{file_directory}/output/', exist_ok=True)
        logging.debug(f"Directory created: {file_directory}")
    else:
        logging.debug(f"Directory already exist: {file_directory}")
    
    return filename, file_directory

def step_1_preprocess(input_file, filename, file_directory):
    """
    Step 1 of the Pipeline:
    -----------------------
    This function preprocesses the input file by parsing, chunking, and categorizing its elements. 
    It then saves intermediate results in various formats and sets up a vector store for the document.
    	
    Params:
        input_file (str): The path to the input file to be processed.
        filename (str): The filename of the input datae.
        file_directory (str): The path to the directory containing the input data and the result folder.
    Returns:
        tuple: A tuple containing:
            - elements_chunked (list): The list of chunked elements.
            - namespace (str): The namespace used in the vector store.
            - vectorstore (PineconeVectorStore): The vector store object containing the processed data.
    """
    # Parse the input file with unstructured API
    elements = unstructured_api_wo_chunking(file=input_file)

    # OPTIONAL: persist/save results as pickle
    with open(f"{file_directory}/by-products/partition_{filename}.pkl", 'wb') as f:
        pickle.dump(elements, f)

    # Persist partition in Excel and CSV
    elements_df = convert_to_dataframe(elements)
    elements_df.to_excel(f"{file_directory}/by-products/partition_wo-chunking_{filename}.xlsx")
    elements_df.to_csv(f"{file_directory}/by-products/partition_wo-chunking_{filename}.csv")
    
    try:
        if CHUNK:
            elements = chunk_elements(elements)
            logging.debug("Elements chunked successfully")

        table_elements, image_elements, text_elements = categorize_elements(elements)
        logging.debug("Elements categorized successfully")

        texts = [i.text for i in text_elements]
        tables = [i.text_as_html for i in table_elements]

        namespace = normalize_filename(f"{filename}_chunked_orig-text")
        vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=EMBEDDINGS, namespace=namespace)
        logging.debug("Vector store initialized successfully")

        index = vectorstore.get_pinecone_index(INDEX_NAME)
        logging.debug("Index retrieved successfully")

        if namespace_exists(index=index, namespace=namespace):
            vectorstore.delete(delete_all=True)
            logging.debug("Existing vectors deleted successfully")

        add_documents_to_vectorstore(vectorstore, texts, text_elements)
        logging.debug("Text summaries added successfully")

        add_documents_to_vectorstore(vectorstore, tables, table_elements)
        logging.debug("Table summaries added successfully")

        return elements, namespace, vectorstore

    except OSError as e:
        logging.error(f"OSError encountered: {e}")
        raise

def step_2_retrieve(elements_chunked, namespace, vectorstore, file_directory, query=COMBINED_QUERY, sim_threshold=0.75):
    """
    Step 2 of the Pipeline:
    -----------------------
    This function retrieves elements from the vector store based on the predefined query and similarity threshold.
    
    Params:
        elements_chunked (list): The list of chunked elements.
        namespace (str): The namespace used in the vector store.
        vectorstore (PineconeVectorStore): The vector store object containing the processed data.
        file_directory (str): The path to the directory containing the input data and the result folder.
        query (str, optional): The query used to retrieve elements. Defaults to COMBINED_QUERY.
        sim_threshold (float, optional): The similarity threshold for element retrieval. Defaults to 0.75.
    Returns:
        retrieved_elements (dict): A dictionary of retrieved elements.
    Raises:
        OSError: If an error occurs while accessing the file system.
    """
    try:
        # Dynamic query with similarity threshold
        k_static = len(elements_chunked) # search entire set of partitions
        retrieved_elements = get_docs_from_vectorstore(vectorstore=vectorstore, index_name=INDEX_NAME, namespace=namespace, query=query, embedding=EMBEDDINGS, k=k_static, score_threshold=sim_threshold)
        logging.debug("Elements retrieved successfully")
    
        # Persist retrieved elements in JSON Object
        with open(f"{file_directory}/by-products/retrieved_elements_{namespace}.json", 'w') as f:
            json.dump(retrieved_elements, f)
        logging.debug("Retrieved elements persisted successfully")

        return retrieved_elements

    except OSError as e:
        logging.error(f"OSError encountered: {e}")
        raise

def step_3_extract_quotes(retrieved_elements, file_directory, namespace):
    """
    Step 3 of the Pipeline:
    -----------------------
    This function extracts quotes from the retrieved elements using a language model and saves the results in a JSON file.
        
    Params:
        retrieved_elements (dict): A dictionary of elements from which quotes are to be extracted.
        file_directory (str): The path to the directory containing the input data and the result folder.
        namespace (str): A namespace identifier used for the vector store.
    Returns:
        dict: A dictionary containing the extracted quotes.
    Raises:
        OSError: If an error occurs while accessing the file system.
    """
    try:
        # Extraction of quotes from retrieved documents using LLM
        quotes_dict = get_quotes(TEXT_MODEL, retrieved_elements)
        logging.debug("Quotes extracted successfully")
            
        #  Persist results in JSON Object
        with open(f"{file_directory}/by-products/extracted_quotes_{TEXT_MODEL.model_name}_{namespace}.json", 'w') as f:
            json.dump(quotes_dict, f)
        logging.debug("Quotes persisted successfully")

        return quotes_dict

    except OSError as e:
        logging.error(f"OSError encountered: {e}")
        raise

def step_4_classify_quotes(quotes_dict, file_directory, namespace):
    """
    Step 4 of the Pipeline:
    -----------------------
    This function classifies the extracted quotes into predefined categories.
    
    Params:
        quotes_dict (dict): A dictionary containing the previously extracted quotes.
        file_directory (str): The path to the directory containing the input data and the result folder.
        namespace (str): A namespace identifier used for the vector store.
    Returns:
        pd.DataFrame : A pandas DF containing the classified quotes.
    """
    try:
        # Classification of quotes into predefined categories
        classified_quotes = tagging_classifier_quotes(quotes_dict=quotes_dict, llm=TEXT_MODEL, fewshot=FEWSHOT)
        logging.debug("Quotes classified successfully")

        #  Persist results in Excel and CSV{TEXT_MODEL.model_name}_{namespace}
        classified_quotes.to_excel(f"{file_directory}/by-products/zero-shot-tagging_{TEXT_MODEL.model_name}_{namespace}.xlsx")
        classified_quotes.to_csv(f"{file_directory}/by-products/zero-shot-tagging_{TEXT_MODEL.model_name}_{namespace}.csv")
        
        return classified_quotes

    except OSError as e:
        logging.error(f"OSError encountered: {e}")
        raise

def postprocess_results(filename, output_df, file_directory, namespace):
    """
    Post-processing of Results:
    -----------------------
    This function performs several postprocessing steps on the output DataFrame from the pipeline.
    It generates highlighted PDFs, and saves specific subsets of the data (targets, mitigation measures, 
    and adaptation measures) to both Excel and CSV files.

    Params:
        output_df (pd.DataFrame): The DataFrame containing the results of the pipeline.
        file_directory (str): The directory where the output files will be saved.
        namespace (str): A namespace string used to differentiate the output files.
    Returns:
        targets_output_df (pd.DataFrame): DataFrame containing the filtered targets data.
        mitigation_output_df (pd.DataFrame): DataFrame containing the filtered mitigation measures data.
        adaptation_output_df (pd.DataFrame): DataFrame containing the filtered adaptation measures data.
    """

    # highlight quotes
    create_highlighted_pdf(filename, 
                            quotes=output_df['quote'],
                            output_path=f"{file_directory}/output/highlighted_{namespace}.pdf",
                            df = output_df
                            )

    #  targets
    targets_output_df = output_df[output_df['target']=='True']
    targets_output_df=targets_output_df[['quote', 'page', 'target_labels']]
    #targets_output_df.to_excel(f"{file_directory}/output/targets_{namespace}.xlsx")
    targets_output_df.to_csv(f"{file_directory}/output/targets_{namespace}.csv")
    
    #  mitigation
    mitigation_output_df = output_df[output_df['mitigation_measure']=='True']
    mitigation_output_df = mitigation_output_df[['quote', 'page', 'measure_labels']]
    #mitigation_output_df.to_excel(f"{file_directory}/output/mitigation_{namespace}.xlsx")
    mitigation_output_df.to_csv(f"{file_directory}/output/mitigation_{namespace}.csv")
    
    #  adaptation
    adaptation_output_df = output_df[output_df['adaptation_measure']=='True']
    adaptation_output_df = adaptation_output_df[['quote', 'page', 'measure_labels']]
    #adaptation_output_df.to_excel(f"{file_directory}/output/adaptation_{namespace}.xlsx")
    adaptation_output_df.to_csv(f"{file_directory}/output/adaptation_{namespace}.csv")

    #  save to single Excel file
    #  create an Excel writer object
    with pd.ExcelWriter(f"{file_directory}/output/results_{namespace}.xlsx") as writer:
    
        # use to_excel function and specify the sheet_name and index 
        # to store the dataframe in specified sheet
        targets_output_df.to_excel(writer, sheet_name="Targets", index=False)
        mitigation_output_df.to_excel(writer, sheet_name="Mitigation", index=False)
        adaptation_output_df.to_excel(writer, sheet_name="Adaptation", index=False)

    return targets_output_df, mitigation_output_df, adaptation_output_df

if __name__ == '__main__':
    logger.info("\n\n========================= Start App =========================\n")
    app.run(debug=True)
