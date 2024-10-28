# -*- coding: utf-8 -*-
"""

Created on Fri May  3 12:05:13 2024

@author: becker_nic

Complete Pipeline in one file.

From PDF ingestion to classification output

Step 1: Preprocessing
    - Send PDF to Unstructured
    - Postprocess Results to Element Objects
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
#import fitz
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
summary = False

#TODO: Check with Marion/Nikola/Tim
keywords = ["Transport",
            "Energy",
            "Net Zero",
            "Mitigation",
            "Adaptation",
            "Conditional",
            "Target",
            "Measure"
            ]

keywords_ALT = ["Transport Sector",
                "Energy Sector",
                "Net Zero Target",
                "Mitigation Measure",
                "Adaptation Measure",
                "Conditional"
                ]

token_query =  "transport  vehicles  emissions  BA  electric U  GH  reduction  public G  compared  scenario  transportation  reduce  CO  levels  net es  zero  fuel  vehicle  em  passenger  road  mobility  veh  target  new idad CO  Mt  car ision  transporte  rail  cars  fleet  buses  fuels  traffic  efficiency ículos ar ct e  gas  greenhouse  redu  freight  d  l  share  km o  bio  achieve os  elé els  hydrogen  urban  infrastructure  electr The  hybrid  relative  charging  neutrality eq  é ici )  least  total ado  emission  vé  standards én  aims  e  ambition ’  modes il  carbon  shift as  neutral fu  bus  EV  ré  mov  condition hic  sales  million cción  inter  año  modal  maritime  system  diesel  público  kt  network ules  alternative  cities  percent  heavy re  conditional  Transport  improvement -em  Electric RT  level  use nel  transit  roads  in  light ibles  energ  year rica  goal  aviation  per missions  long  powered  European  consumption arbon ric  lanes  vo  part  walking  sharing  rapport ación  t  bicycle  motor  stations  infra  s duction ov a To  sc  railways  cent  private ías  reductions ). ual r  achieved ada -m condition  élect  ef"

combined_query =  "transport  vehicles  emissions  BA  electric U  GH  reduction  public G  compared  scenario  transportation  reduce  CO  levels  net es  zero  fuel  vehicle  em  passenger  road  mobility  veh  target  new idad CO  Mt  car ision  transporte  rail  cars  fleet  buses  fuels  traffic  efficiency ículos ar ct e  gas  greenhouse  redu  freight  d  l  share  km o  bio  achieve os  elé els  hydrogen  urban  infrastructure  electr The  hybrid  relative  charging  neutrality eq  é ici )  least  total ado  emission  vé  standards én  aims  e  ambition ’  modes il  carbon  shift as  neutral fu  bus  EV  ré  mov  condition hic  sales  million cción  inter  año  modal  maritime  system  diesel  público  kt  network ules  alternative  cities  percent  heavy re  conditional  Transport  improvement -em  Electric RT  level  use nel  transit  roads  in  light ibles  energ  year rica  goal  aviation  per missions  long  powered  European  consumption arbon ric  lanes  vo  part  walking  sharing  rapport ación  t  bicycle  motor  stations  infra  s duction ov a To  sc  railways  cent  private ías  reductions ). ual r  achieved ada -m condition  élect  ef Transport Energy Net Zero Mitigation Adaptation Conditional Target Measure"

#  Vectorstore index name and namespace
index_name = "ndc-summaries"

## Configuration
#load_dotenv()

#  Text model - Azure OpenAI - currently gpt-4-0613 --> https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#gpt-4-and-gpt-4-turbo-models
text_model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    model_version='0613',
    temperature=0
)
#text_model = ChatOpenAI(temperature=0, model='gpt-4o')

#  Vision model
vision_model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    model_version='0613',
    temperature=0
)
#vision_model = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=1024)


#  Embedding function
embeddings = AzureOpenAIEmbeddings(openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                   deployment = 'embeds'
                                   )



#%%
@track_emissions
def main():
    
    total_cost = 0
    total_tokens = 0
    
    # Streamlit application
    st.title("Climate Policy Miner :globe_with_meridians::page_with_curl::pick:") 
    
    try:
        ## STEP 0: Get file
        st.write("### Input")
        filename = set_file(streamlit=True)
        file_directory = os.path.splitext(ntpath.basename(filename))[0]
        logger.info(f'Pipeline execution for file: {ntpath.basename(filename)}')
        
        st.write("Is the file already pre-processed?")
        
        if st.button("No"):
            #TODO - insert estimation of runtume and cost
            st.info(f'Analyzing "{ntpath.basename(filename)}" ... Get yourself some coffee, this might take a while.')            
            ## STEP 1: Preprocessing
            prep_text = f"Preprocessing {ntpath.basename(filename)}"
            prep_bar = st.progress(0, text=prep_text)
            prep_time = t.time()
            with get_openai_callback() as cb:
                
                #  Send PDF to Unstructured
                start_time = t.time()
                elements = unstructured_api_wo_chunking(file=filename)
                s = t.time()-start_time
                logger.info(f'Partitioning took {s} sec.')
                
                prep_bar.progress(50, text=prep_text + ' - Partitioning...')
                
                #  OPTIONAL: persist/save results as pickle
                with open(f"{file_directory}/by-products/partition_{file_directory}.pkl", 'wb') as f:
                    pickle.dump(elements, f)
                
                #  Persist partition in Excel and CSV
                elements_df = convert_to_dataframe(elements)
                elements_df.to_excel(f"{file_directory}/by-products/partition_wo-chunking_{file_directory}.xlsx")
                elements_df.to_csv(f"{file_directory}/by-products/partition_wo-chunking_{file_directory}.csv")
                
                ################ Checkpoint ###################
                #  Load elements from pickle
                with open(f"{file_directory}/by-products/partition_{file_directory}.pkl", 'rb') as f:
                    elements=pickle.load(f)
                ###############################################
                                    
                #  Chunk elements
                elements_chunked = chunk_elements(elements)
                
                prep_bar.progress(75, text=prep_text + ' - Chunking...')
                
                #  Postprocess Results to Element Objects and split by Element type
                table_elements, image_elements, text_elements = categorize_elements(elements_chunked)
                texts = [i.text for i in text_elements]
                tables = [i.text_as_html for i in table_elements]
                
                if summary:
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
                        
                    ################ Checkpoint ###################
                    #  Postprocess Results to Element Objects and split by Element type
                    table_elements, image_elements, text_elements = categorize_elements(elements)
                    #  Load summaries
                    with open(f"{file_directory}/by-products/text_summaries_{file_directory}.pkl", 'rb') as f:
                        text_summaries=pickle.load(f)
                    with open(f"{file_directory}/by-products/table_summaries_{file_directory}.pkl", 'rb') as f:
                        table_summaries=pickle.load(f)
                    ###############################################                
                
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
                
                prep_bar.progress(90, text=prep_text + ' - Loading to VectorStore...')
                    
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
                    
                    prep_bar.progress(100, text=prep_text + ' - Loading to VectorStore...')
                
                logger.info(f"Preprocessing Cost (USD): ${format(cb.total_cost, '.6f')}")
                logger.info(f"Preprocessing Tokens: {cb.total_tokens}")
                total_cost = total_cost + cb.total_cost
                total_tokens = total_tokens + cb.total_tokens
                
            prep_time = t.time() - prep_time
            logger.info(f"Preprocessing took {prep_time} seconds.")
        
        if st.button("Yes"):
            ## STEP 2: Document Retrieval
            retr_time = t.time()
            with get_openai_callback() as cb:
                
                retr_text = "Retrieving Elements from VectorStore..."
                retr_bar = st.progress(0, text=retr_text)
                
                ################### CHECKPOINT ####################
                namespace = normalize_filename(f"{file_directory}_chunked_orig-text") # check if namespace is ASCII-printable
                vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings, namespace=namespace)
                
                #  Load elements from pickle
                with open(f"{file_directory}/by-products/partition_{file_directory}.pkl", 'rb') as f:
                    elements=pickle.load(f)
                    
                #  Chunk elements
                elements_chunked = chunk_elements(elements)
                ###################################################
                
                retr_bar.progress(25, text=retr_text)
                
                
                #  dynamic query with similarity threshold
                k_static = len(elements_chunked) # search entire set of partitions
                rel_docs = get_docs_from_vectorstore(vectorstore=vectorstore, index_name=index_name, namespace=namespace, query=combined_query, embedding=embeddings, k=k_static, score_threshold=0.75)
                
                retr_bar.progress(50, text=retr_text)
                
                #  Persist results in JSON Object
                with open(f"{file_directory}/by-products/retrieved_documents_{file_directory}.json", 'w') as f:
                    json.dump(rel_docs, f)
                    
                retr_bar.progress(100, text=retr_text)
                
                logger.info(f"Document Retrieval Cost (USD): ${format(cb.total_cost, '.6f')}")
                logger.info(f"Document Retrieval Tokens: {cb.total_tokens}")
                total_cost = total_cost + cb.total_cost
                total_tokens = total_tokens + cb.total_tokens
                
            retr_time = t.time() - retr_time
            logger.info(f"Document Retrieval took {retr_time} seconds.")
            
            ## STEP 3: Quote extraction
            quot_text = f"Extracting Quotes..."
            quot_bar = st.progress(0, text=quot_text)
            quot_time = t.time()
            with get_openai_callback() as cb:
                
                quotes_dict = get_quotes(text_model, rel_docs)
                
                quot_bar.progress(50, text=quot_text)
                
                #  Persist results in JSON Object
                with open(f"{file_directory}/by-products/extracted_quotes_{text_model.model_name}_{file_directory}.json", 'w') as f:
                    json.dump(quotes_dict, f)
                    
                quot_bar.progress(100, text=quot_text)
            
                logger.info(f"Quote Extraction Cost (USD): ${format(cb.total_cost, '.6f')}")
                logger.info(f"Quote Extraction Tokens: {cb.total_tokens}")
                total_cost = total_cost + cb.total_cost
                total_tokens = total_tokens + cb.total_tokens
                
            quot_time = t.time() - quot_time
            logger.info(f"Quote Extraction took {quot_time} seconds.")
            
            ################### CHECKPOINT ####################
            with open(f"{file_directory}/by-products/extracted_quotes_{text_model.model_name}_{file_directory}.json", 'r') as f:
                quotes_dict = json.load(f)
            ###################################################
            
            ## STEP 4: Classification
            clas_text = f"Classifying Quotes..."
            clas_bar = st.progress(0, text=clas_text)
            clas_time = t.time()
            with get_openai_callback() as cb:
            
                #  Few-Shot Tagging Classifier
                output_df = tagging_classifier_quotes(quotes_dict=quotes_dict, llm=text_model, fewshot=True)
                clas_bar.progress(50, text=clas_text)
                
                #  Persist results in Excel and CSV
                output_df.to_excel(f"{file_directory}/by-products/zero-shot-tagging_{text_model.model_name}_{file_directory}.xlsx")
                output_df.to_csv(f"{file_directory}/by-products/zero-shot-tagging_{text_model.model_name}_{file_directory}.csv")
                
                clas_bar.progress(100, text=clas_text)
                
                logger.info(f"Classification Cost (USD): ${format(cb.total_cost, '.6f')}")
                logger.info(f"Classification Tokens: {cb.total_tokens}")
                total_cost = total_cost + cb.total_cost
                total_tokens = total_tokens + cb.total_tokens
                
            clas_time = t.time() - clas_time
            logger.info(f"Classification took {clas_time} seconds.")
            
            post_text = f"Postprocessing Results..."
            post_bar = st.progress(0, text=post_text)
            
            #  highlight quotes
            create_highlighted_pdf(filename, 
                                    quotes=output_df['quote'],
                                    output_path=f"{file_directory}/output/highlighted_{ntpath.basename(filename)}",
                                    df = output_df)
            
            post_bar.progress(30, text=post_text)
            
            #  targets
            targets_output_df = output_df[output_df['target']=='True']
            targets_output_df=targets_output_df[['quote', 'page', 'target_labels']]
            targets_output_df.to_excel(f"{file_directory}/output/targets_{file_directory}.xlsx")
            targets_output_df.to_csv(f"{file_directory}/output/targets_{file_directory}.csv")
            post_bar.progress(50, text=post_text)
            
            #  mitigation
            mitigation_output_df = output_df[output_df['mitigation_measure']=='True']
            mitigation_output_df = mitigation_output_df[['quote', 'page', 'measure_labels']]
            mitigation_output_df.to_excel(f"{file_directory}/output/mitigation_{file_directory}.xlsx")
            mitigation_output_df.to_csv(f"{file_directory}/output/mitigation_{file_directory}.csv")
            post_bar.progress(75, text=post_text)
            
            #  adaptation
            adaptation_output_df = output_df[output_df['adaptation_measure']=='True']
            adaptation_output_df = adaptation_output_df[['quote', 'page', 'measure_labels']]
            adaptation_output_df.to_excel(f"{file_directory}/output/adaptation_{file_directory}.xlsx")
            adaptation_output_df.to_csv(f"{file_directory}/output/adaptation_{file_directory}.csv")
            post_bar.progress(100, text=post_text)
            
            
            #  celebration
            st.balloons()
            
            #  show results
            st.write("### Results")
            st.write("#### Targets")
            st.write(targets_output_df)
            st.write("#### Mitigation")
            st.write(mitigation_output_df)
            st.write("#### Adaptation")
            st.write(adaptation_output_df)
            
            #  metrics
            st.write("### Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Total Tokens", value=f"{total_tokens}")
            col2.metric(label="Total Costs", value=f"${format(total_cost, '.6f')}")
            col3.metric(label="Total Runtime", value=f"{prep_time+retr_time+quot_time+clas_time} sec.")
        
    except Exception as e:
        logging.exception(f"Pipeline exit with an error: {e}")
        
    finally:
    
        logger.info(f"Total Cost (USD): ${format(total_cost, '.6f')}")
        logger.info(f"Total Tokens: {total_tokens}")
    
    
if __name__ == "__main__":
    
    logger.info("\n\n========================= Start Pipeline =========================\n")
    total_time = t.time()
    main()
    total_time = t.time() - total_time
    logger.info(f"Pipeline took {total_time} seconds.")
    

#%% Test section

