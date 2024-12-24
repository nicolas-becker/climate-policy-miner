# -*- coding: utf-8 -*-
"""

Created on Fri May  3 12:05:13 2024

@author: becker_nic

Complete Pipeline in one file.

From PDF ingestion to classification output

Step 1: Preprocessing
    - Send PDF to Unstructured
    - Postprocess results to Element Objects
    - Split Elements by Element type
    - Summarize contents
    - Pass Summaries to VectorStore with Metadata

Step 2: Document Retrieval
    - Define Keyword list
    - Pass keywords to perform Conext Search on VectorStore
    - Retrieve top-k matching vectors with metadata, incl. original content
    - Save as JSON Object
    
Step 3: Quote Extraction
    - Load JSON Object with retrieved documents
    - Apply quote extraction chain on original content
    - Verify quotes to avoid halluscination
    - Add quotes to Copy of JSON Object
    - Save copy of JSON Object with quotes
    
Step 4: Quote Classification
    - Load JSON with quotes
    - Apply Classification chain on quotes
    - Save results in DataFrame object
    - Postprocess Dataframe to adapt output
    
"""

import pickle 
import ntpath
import time as t
import os
import json
import fitz
import logging
import warnings
import streamlit as st

from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.document_loaders import OnlinePDFLoader #TODO
from unstructured.staging.base import convert_to_dataframe
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain import globals
from langchain_pinecone import PineconeVectorStore
from langsmith.wrappers import wrap_openai
from langsmith import Client, traceable
from langsmith.evaluation import evaluate
from openai._exceptions import BadRequestError
from langchain_community.callbacks import get_openai_callback
#from openai import Client
#from pinecone import Pinecone
from preprocessing_utils import set_file, unstructured_api_wo_chunking, categorize_elements, summarize_elements, summarize_text, summarize_table, summarize_image, add_documents_to_vectorstore, chunk_elements
from retrieval_utils import get_docs_from_vectorstore
from quotation_utils import get_quotes
from classification_utils import tagging_classifier_quotes
from general_utils import setup_logger, normalize_filename, create_highlighted_pdf, namespace_exists
from codecarbon import EmissionsTracker, track_emissions

# Set verbose
globals.set_debug(True)

# warnings
warnings.filterwarnings('ignore')

# Set logging
logger = setup_logger(__name__, 'logfile.log')

# Auto-trace LLM calls in-context
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "climate_policy_pipeline"


## Params

#  Initial keyword list for query
keywords = ["Transport",
            "Energy",
            "Net Zero",
            "Mitigation",
            "Adaptation",
            "Conditional",
            "Target",
            "Measure"
            ]

#  Inferred query from frequent tokens in database
token_query =  "transport  vehicles  emissions  BA  electric U  GH  reduction  public G  compared  scenario  transportation  reduce  CO  levels  net es  zero  fuel  vehicle  em  passenger  road  mobility  veh  target  new idad CO  Mt  car ision  transporte  rail  cars  fleet  buses  fuels  traffic  efficiency ículos ar ct e  gas  greenhouse  redu  freight  d  l  share  km o  bio  achieve os  elé els  hydrogen  urban  infrastructure  electr The  hybrid  relative  charging  neutrality eq  é ici )  least  total ado  emission  vé  standards én  aims  e  ambition ’  modes il  carbon  shift as  neutral fu  bus  EV  ré  mov  condition hic  sales  million cción  inter  año  modal  maritime  system  diesel  público  kt  network ules  alternative  cities  percent  heavy re  conditional  Transport  improvement -em  Electric RT  level  use nel  transit  roads  in  light ibles  energ  year rica  goal  aviation  per missions  long  powered  European  consumption arbon ric  lanes  vo  part  walking  sharing  rapport ación  t  bicycle  motor  stations  infra  s duction ov a To  sc  railways  cent  private ías  reductions ). ual r  achieved ada -m condition  élect  ef"

#  Combination of token_query and keywords
combined_query =  "transport  vehicles  emissions  BA  electric U  GH  reduction  public G  compared  scenario  transportation  reduce  CO  levels  net es  zero  fuel  vehicle  em  passenger  road  mobility  veh  target  new idad CO  Mt  car ision  transporte  rail  cars  fleet  buses  fuels  traffic  efficiency ículos ar ct e  gas  greenhouse  redu  freight  d  l  share  km o  bio  achieve os  elé els  hydrogen  urban  infrastructure  electr The  hybrid  relative  charging  neutrality eq  é ici )  least  total ado  emission  vé  standards én  aims  e  ambition ’  modes il  carbon  shift as  neutral fu  bus  EV  ré  mov  condition hic  sales  million cción  inter  año  modal  maritime  system  diesel  público  kt  network ules  alternative  cities  percent  heavy re  conditional  Transport  improvement -em  Electric RT  level  use nel  transit  roads  in  light ibles  energ  year rica  goal  aviation  per missions  long  powered  European  consumption arbon ric  lanes  vo  part  walking  sharing  rapport ación  t  bicycle  motor  stations  infra  s duction ov a To  sc  railways  cent  private ías  reductions ). ual r  achieved ada -m condition  élect  ef Transport Energy Net Zero Mitigation Adaptation Conditional Target Measure"

#  Vectorstore index name and namespace
index_name = "ndc-summaries"

#  Load .env file
load_dotenv()

#  Text model - Azure OpenAI - currently gpt-4-0613 --> https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#gpt-4-and-gpt-4-turbo-models
text_model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    model_version='0613',
    temperature=0
)

### if GPT Access is not possible via MS Azure, replace the text_model with the following: 
#text_model = ChatOpenAI(temperature=0, model='gpt-4o')

#  Vision model
vision_model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_VISION"],
    model_version='0613',
    temperature=0
)

### if GPT Access is not possible via MS Azure, replace the vision_model with the following: 
#vision_model = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=1024)


#  Embedding function
embeddings = AzureOpenAIEmbeddings(openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                   deployment = 'embeds'
                                   )



#%% Pipeline
@track_emissions
def main():
    
    total_cost = 0
    total_tokens = 0
    
    try:
        ## STEP 0: Get file
        filename = set_file()
        file_directory = os.path.splitext(ntpath.basename(filename))[0]
        logger.info(f'Pipeline execution for file: {ntpath.basename(filename)}')
        
        preprocessing = input('Is the file already pre-processed? [y/n]')
        
        #TODO - insert estimation of runtume and cost
        print(f'\n\n-------------------------\
              \n\nAnalyzing "{ntpath.basename(filename)}" ...\n\nGet yourself some coffee, this might take a while.')
        
        if (preprocessing == 'n') | (preprocessing == 'no'):
            
            ## STEP 1: Preprocessing
            print('\nStart Preprocessing...')
            prep_time = t.time()
            with get_openai_callback() as cb:
                
                #  Send PDF to Unstructured
                start_time = t.time()
                elements = unstructured_api_wo_chunking(file=filename)
                s = t.time()-start_time
                print(f'\n\nPartitioning took {s} sec.')
                logger.info(f'Partitioning took {s} sec.')
                
                #  OPTIONAL: persist/save results as pickle
                with open(f"{file_directory}/by-products/partition_{file_directory}.pkl", 'wb') as f:
                    pickle.dump(elements, f)
                
                """
                ################ Checkpoint ###################
                #  Load elements from pickle
                with open(f"{file_directory}/by-products/partition_{file_directory}.pkl", 'rb') as f:
                    elements=pickle.load(f)
                ###############################################
                """
                
                #  Persist partition in Excel and CSV
                elements_df = convert_to_dataframe(elements)
                elements_df.to_excel(f"{file_directory}/by-products/partition_wo-chunking_{file_directory}.xlsx")
                elements_df.to_csv(f"{file_directory}/by-products/partition_wo-chunking_{file_directory}.csv")
                    
                #  Chunk elements
                elements_chunked = chunk_elements(elements)
                
                #  Postprocess Results to Element Objects and split by Element type
                table_elements, image_elements, text_elements = categorize_elements(elements_chunked)
                texts = [i.text for i in text_elements]
                tables = [i.text_as_html for i in table_elements]
                
                """ Summaries currently not used

                #  Summarize Texts
                start_time = t.time()
                text_summaries = summarize_text(elements=texts, model=text_model)
                s = t.time()-start_time
                print(f'\n\nSummarization of text elements took {s} sec.')
                logger.info(f'Summarization of text elements took {s} sec.')
                logger.info(f"Summarization of text elements Cost (USD): ${format(cb.total_cost, '.6f')}")
                logger.info(f"Summarization of text elements Tokens: {cb.total_tokens}")
                
                #  Summarize Tables
                start_time = t.time()
                table_summaries = summarize_table(elements=tables, model=text_model)
                s = t.time()-start_time
                print(f'\n\nSummarization of table elements took {s} sec.')
                logger.info(f'Summarization of table elements took {s} sec.')
                logger.info(f"Summarization of table elements Cost (USD): ${format(cb.total_cost, '.6f')}")
                logger.info(f"Summarization of table elements Tokens: {cb.total_tokens}")
                
                #  Persist summaries
                with open(f"{file_directory}/text_summaries_{file_directory}.pkl", 'wb') as f:
                    pickle.dump(text_summaries, f)
                with open(f"{file_directory}/table_summaries_{file_directory}.pkl", 'wb') as f:
                    pickle.dump(table_summaries, f)   
                """
                """
                ################ Checkpoint ###################
                #  Chunk elements
                elements_chunked = chunk_elements(elements)
                #  Postprocess Results to Element Objects and split by Element type
                table_elements, image_elements, text_elements = categorize_elements(elements_chunked)
                #  Load summaries
                with open(f"{file_directory/by-products}/text_summaries_{file_directory}.pkl", 'rb') as f:
                    text_summaries=pickle.load(f)
                with open(f"{file_directory/by-products}/table_summaries_{file_directory}.pkl", 'rb') as f:
                    table_summaries=pickle.load(f)
                ###############################################                
                """
                #  Set up vector store with namespace dedicated to current document
                #  https://app.pinecone.io/organizations/-Nux-qonUJ0fgF6B5oAb/projects/85bf2d16-08d0-412f-96e7-bed39cd531bf/indexes/ndc-summaries/namespaces
                # TODO: check if already exists and delete existing entries first
                namespace = normalize_filename(f"{file_directory}_chunked_orig-text") # check if namespace is ASCII-printable
                vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
                #  drop existing vectors for the docuemnt to avoid duplicates
                index = vectorstore.get_pinecone_index(index_name)
                if namespace_exists(index=index, namespace=namespace):
                    vectorstore.delete(delete_all = True)
                    logger.info(f'Existing namespace "{namespace}" was overwritten.')
                    print(f'Existing namespace "{namespace}" was overwritten.')
                else:
                    print(f'New namespace "{namespace}" is created in index "{index_name}".')
                    
                #  Add text summaries
                add_documents_to_vectorstore(vectorstore, texts, text_elements)
                #  Add table summaries
                add_documents_to_vectorstore(vectorstore, tables, table_elements)
                
                #  Process Images
                if vision_model.model_name.startswith('gpt-4'):
                    # TODO: Extract Images to folder - Alternative to Unstructured (installed version - requires C++ Builder)
                    #  retrieve base64-encoded images from folder - CURRENTLY NOT USED - unstructured api returns base64 encoding in metadata directly
                    #images = retrieve_images_from_folder(image_folder)
                    images = [i.image_base64 for i in image_elements]
                    
                    #  summarize images
                    start_time = t.time()
                    image_summaries = []
                    for i, ie in enumerate(images):
                        try:
                            summary = summarize_image(encoded_image=ie, model=vision_model)
                            image_summaries.append(summary)
                        except BadRequestError:
                            image_summaries.append('Image was not processed due to a BadRequestError')
                        #print(f"\n\n{i + 1}th element of images processed.")
                    s = t.time()-start_time
                    print(f'\n\nSummarization of imagees took {s} sec.')
                    
                    #  persist image summaries
                    with open(f"{file_directory}/by-products/image_summaries_{file_directory}.pkl", 'wb') as f:
                        pickle.dump(image_summaries, f)
                    
                    #  Add image summaries to vector store
                    add_documents_to_vectorstore(vectorstore, image_summaries, image_elements) 
                
                print(f"Preprocessing Cost (USD): ${format(cb.total_cost, '.6f')}")
                logger.info(f"Preprocessing Cost (USD): ${format(cb.total_cost, '.6f')}")
                logger.info(f"Preprocessing Tokens: {cb.total_tokens}")
                total_cost = total_cost + cb.total_cost
                total_tokens = total_tokens + cb.total_tokens
                
            prep_time = t.time() - prep_time
            logger.info(f"Preprocessing took {prep_time} seconds.")
                            
            
        ################### CHECKPOINT ####################
        
        #  load vectorstore
        namespace = normalize_filename(f"{file_directory}_chunked_orig-text") # check if namespace is ASCII-printable
        vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings, namespace=namespace)
        
        #  Load elements from pickle
        with open(f"{file_directory}/by-products/partition_{file_directory}.pkl", 'rb') as f:
            elements=pickle.load(f)
        #  Chunk elements
        elements_chunked = chunk_elements(elements)
        #  Postprocess Results to Element Objects and split by Element type
#        table_elements, image_elements, text_elements = categorize_elements(elements_chunked)
        #  Load summaries
#            with open(f"{file_directory}/by-products/text_summaries_{file_directory}.pkl", 'rb') as f:
#                text_summaries=pickle.load(f)
#            with open(f"{file_directory}/by-products/table_summaries_{file_directory}.pkl", 'rb') as f:
#                table_summaries=pickle.load(f)
        ###################################################            
        
        ## STEP 2: Document Retrieval
        print('Start Document Retrieval\n')
        retr_time = t.time()
        with get_openai_callback() as cb:
            
            #  load vectorstore
            #namespace = normalize_filename(f"{file_directory}_chunked_orig-text") # check if namespace is ASCII-printable
            #vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings, namespace=namespace)
            
            #  dynamic query with similarity threshold
            sim_threshold = 0.75
            k_static = len(elements_chunked) # search entire set of partitions
            rel_docs = get_docs_from_vectorstore(vectorstore=vectorstore, index_name=index_name, namespace=namespace, query=combined_query, embedding=embeddings, k=k_static, score_threshold=sim_threshold)

            #  Persist results in JSON Object
            with open(f"{file_directory}/by-products/retrieved_documents_{file_directory}.json", 'w') as f:
                json.dump(rel_docs, f)
                
            print(f"Document Retrieval Cost (USD): ${format(cb.total_cost, '.6f')}")
            logger.info(f"Document Retrieval Cost (USD): ${format(cb.total_cost, '.6f')}")
            logger.info(f"Document Retrieval Tokens: {cb.total_tokens}")
            total_cost = total_cost + cb.total_cost
            total_tokens = total_tokens + cb.total_tokens
            
        retr_time = t.time() - retr_time
        logger.info(f"Document Retrieval took {retr_time} seconds.")
        
        ## STEP 3: Quote extraction
        print('Start Quote Extraction\n')
        quot_time = t.time()
        with get_openai_callback() as cb:
            quotes_dict = get_quotes(text_model, rel_docs)
            
            #  Persist results in JSON Object
            with open(f"{file_directory}/by-products/extracted_quotes_{text_model.model_name}_{file_directory}.json", 'w') as f:
                json.dump(quotes_dict, f)
        
            print(f"Quote Extraction Cost (USD): ${format(cb.total_cost, '.6f')}")
            logger.info(f"Quote Extraction Cost (USD): ${format(cb.total_cost, '.6f')}")
            logger.info(f"Quote Extraction Tokens: {cb.total_tokens}")
            total_cost = total_cost + cb.total_cost
            total_tokens = total_tokens + cb.total_tokens
            
        quot_time = t.time() - quot_time
        logger.info(f"Quote Extraction took {quot_time} seconds.")
        
        """
        ################### CHECKPOINT ####################
        with open(f"{file_directory}/by-products/extracted_quotes_{text_model.model_name}_{file_directory}.json", 'r') as f:
            quotes_dict = json.load(f)
        ###################################################
        """
        
        ## STEP 4: Classification
        print('Start Classification\n')
        clas_time = t.time()
        with get_openai_callback() as cb:
            #  Zero-Shot Classifier (Targets)
            """
            start_time=t.time()
            classification_chain = zero_shot_target_classification_chain(llm=text_model)
            output_df = classify_target_quotes(quotes_dict=quotes_dict, chain=classification_chain)
            print(f"\n\nClassification took {t.time()-start_time} sec.")
            """
        
            #  Few-Shot Tagging Classifier
            start_time=t.time()
            output_df = tagging_classifier_quotes(quotes_dict=quotes_dict, llm=text_model, fewshot=True)
            print(f"\n\nClassification took {t.time()-start_time} sec.")
            
            #  Persist results in Excel and CSV
            output_df.to_excel(f"{file_directory}/by-products/zero-shot-tagging_{text_model.model_name}_{file_directory}.xlsx")
            output_df.to_csv(f"{file_directory}/by-products/zero-shot-tagging_{text_model.model_name}_{file_directory}.csv")
            
            print(f"Classification Cost (USD): ${format(cb.total_cost, '.6f')}")
            logger.info(f"Classification Cost (USD): ${format(cb.total_cost, '.6f')}")
            logger.info(f"Classification Tokens: {cb.total_tokens}")
            total_cost = total_cost + cb.total_cost
            total_tokens = total_tokens + cb.total_tokens
            
        clas_time = t.time() - clas_time
        logger.info(f"Classification took {clas_time} seconds.")
            
        #  highlight quotes
        create_highlighted_pdf(filename, 
                            quotes=output_df['quote'],
                            output_path=f"{file_directory}/output/highlighted_{ntpath.basename(filename)}",
                            df = output_df)
        
        #  targets
        targets_output_df = output_df[output_df['target']=='True']
        targets_output_df=targets_output_df[['quote', 'page', 'target_labels']]
        targets_output_df.to_excel(f"{file_directory}/output/targets_{file_directory}.xlsx")
        targets_output_df.to_csv(f"{file_directory}/output/targets_{file_directory}.csv")
        
        #  mitigation
        mitigation_output_df = output_df[output_df['mitigation_measure']=='True']
        mitigation_output_df = mitigation_output_df[['quote', 'page', 'measure_labels']]
        mitigation_output_df.to_excel(f"{file_directory}/output/mitigation_{file_directory}.xlsx")
        mitigation_output_df.to_csv(f"{file_directory}/output/mitigation_{file_directory}.csv")
        
        #  adaptation
        adaptation_output_df = output_df[output_df['adaptation_measure']=='True']
        adaptation_output_df = adaptation_output_df[['quote', 'page', 'measure_labels']]
        adaptation_output_df.to_excel(f"{file_directory}/output/adaptation_{file_directory}.xlsx")
        adaptation_output_df.to_csv(f"{file_directory}/output/adaptation_{file_directory}.csv")
        
        
        #  Append Link to Page for each quote
        #  TODO:
        #  https://pymupdf.readthedocs.io/en/latest/the-basics.html#getting-page-links
        #  Add funtionality to highlight function, to avoid double processing with PyMuPDF
        #  Alternative: 
        #  via Coordinates - https://pymupdf.readthedocs.io/en/latest/app3.html#coordinates
            

    except Exception as e:
        logging.exception(f"Pipeline exit with an error: {e}")
        
    finally:
    
        print(f"\n\nTotal Cost (USD): ${format(total_cost, '.6f')}")
        logger.info(f"Total Cost (USD): ${format(total_cost, '.6f')}")
        logger.info(f"Total Tokens: {total_tokens}")
        print(f"\n\nResults can be found in the folder: '{file_directory}'")
    
    
if __name__ == "__main__":
    
    logger.info("\n\n========================= Start Pipeline =========================\n")
    total_time = t.time()
    main()
    total_time = t.time() - total_time
    print(f"\n\nOverall execution time: {total_time} sec.")
    logger.info(f"Pipeline took {total_time} seconds.")
    

#%% Test section

