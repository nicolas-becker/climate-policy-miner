# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:07:20 2024

@author: becker_nic

PREPROCESSING MODULE
"""

import os
import ntpath
import base64
import urllib.parse
import backoff
import streamlit as st

from dotenv import load_dotenv
from typing import Any, List, Optional
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.document import Document
from openai._exceptions import RateLimitError

from unstructured.partition.api import partition_via_api
from unstructured.chunking.title import chunk_by_title

from langsmith import traceable

from general_utils import download_pdf, setup_logger, is_valid_url

#  Set logging
logger = setup_logger(__name__, 'logfile.log')

#  Load environment variables
load_dotenv()



class Element(BaseModel):
    type: str
    element_id: str 
    text: Any
    languages: List[str]
    page_number: int 
    image_base64: Optional[str] = None
    image_mime_type: Optional[str] = None
    filename: str    
    parent_id: Optional[str] = None
    text_as_html: Optional[str] = None
    coordinates: Optional[dict] = None
    #relevance: bool 

def set_file(streamlit = False):
    """
    Gets Filename from user input

    Returns
    -------
    filename : str
        String containg path to the PDF.

    """
    #TODO: add option of inserting url to pdf --> https://stackoverflow.com/questions/9751197/opening-pdf-urls-with-pypdf
    
    if streamlit:
        # Input field for file path or URL
        filename = st.text_input("Paste the path or URL to the file you would like to be analyzed:").strip('"')
        
        # Button to submit the input
        if st.button("Upload"):
            #  handle URLs - to be tested
            if is_valid_url(filename.strip('"')):    
                filename = handle_url(filename)
                
                # creating image file for document
                if not os.path.exists(f'{os.path.splitext(ntpath.basename(filename))[0]}'):
                    with st.status('Creating folder...'):
                        os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}')
                        os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}/by-products/')
                        os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}/by-products/figures/')
                        os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}/output/')
                
                st.success('\nDocument folder exists or was created.\n')
        
            # test if path exists
            elif os.path.exists(filename):
                
                # creating image file for document
                if not os.path.exists(f'{os.path.splitext(ntpath.basename(filename))[0]}'):
                    with st.status('Creating folder...'):
                        os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}')
                        os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}/by-products/')
                        os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}/by-products/figures/')
                        os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}/output/')
                
                st.success('\nDocument folder exists or was created.\n')
                
            else:
                st.error('The specified file does NOT exist. Please try again.')
        

    else:
        # console input
        print("Paste the path or URL to the file you would like to be analyzed:")    
        filename = input().strip('"')
        
        #filename = rstr(filename)
    
        #  handle URLs - to be tested
        if is_valid_url(filename.strip('"')):    
            filename = handle_url(filename)
            
            # creating image file for document
            if not os.path.exists(f'{os.path.splitext(ntpath.basename(filename))[0]}'):
                print('\n\nCreating folder...')
                os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}')
                os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}/by-products/')
                os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}/by-products/figures/')
                os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}/output/')
            
            print('\nDocument folder exists or was created.\n')
    
        # test if path exists
        elif os.path.exists(filename):
            
            # creating image file for document
            if not os.path.exists(f'{os.path.splitext(ntpath.basename(filename))[0]}'):
                print('\n\nCreating folder...')
                os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}')
                os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}/by-products/')
                os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}/by-products/figures/')
                os.makedirs(f'{os.path.splitext(ntpath.basename(filename))[0]}/output/')
            
            print('\nDocument folder exists or was created.\n')
            
        else:
            print('The specified file does NOT exist. Please try again.')
                
    return filename


def handle_url(url : str):
    """
    Gets Filename from user input URL

    Returns
    -------
    filename : str
        String containg path to the PDF.

    """
    
    url_parsed = urllib.parse.unquote(url)
     
    # Download the PDF
    pdf_path = f"{ntpath.basename(url_parsed)}/{ntpath.basename(url_parsed)}"
    download_pdf(url, pdf_path)
                
    return pdf_path
      
  
def unstructured_api_wo_chunking(file):
    """
    Partitions the PDF file via unstructureds API
    --> https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/partition/api.py
    Without chunking, the original entire range of possible element types 
    (https://unstructured-io.github.io/unstructured/introduction/overview.html#id1) 
    is includes in the partition.
    --> https://github.com/Unstructured-IO/unstructured-python-client/blob/main/docs/models/shared/chunkingstrategy.md
    General Partition Parameters:
    --> https://github.com/Unstructured-IO/unstructured-python-client/blob/main/docs/models/shared/partitionparameters.md    

    Parameters
    ----------
    file : str
        Path to PDF file.

    Returns
    -------
    elements : list of Element Objects
        Contains the partition into the different info representations present in the PDF document.

    """
    elements = partition_via_api(
        filename=file, 
        api_key=os.environ["UNSTRUCTURED_API_KEY"], 
        strategy="hi_res", 
        coordinates = "true",  # return bounding box coordinates for each element extracted via OCR
        pdf_infer_table_structure="true",
        extract_image_block_types=["Image", "Table"] # The types of elements to extract, for use in extracting image blocks as base64 encoded data stored in metadata fields
     #   unique_element_ids=True #assign UUIDs to element IDs, which guarantees their uniqueness (useful when using them as primary keys in database). Otherwise a SHA-256 of element text is used.     
     )

    return elements

def unstructured_api_chunking_bytitle(file):
    """
    Partitions the PDF file via unstructureds API
    --> https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/partition/api.py
    General Partition Parameters:
    --> https://github.com/Unstructured-IO/unstructured-python-client/blob/main/docs/models/shared/partitionparameters.md    
    Problem: 
    --> output does not include images
    --> output does not include parent ids

    Parameters
    ----------
    file : str
        Path to PDF file.

    Returns
    -------
    elements : list of Element Objects
        Contains the partition into the different info representations present in the PDF document.

    """
    elements_by_title = partition_via_api(
        filename=file, 
        api_key=os.environ["UNSTRUCTURED_API_KEY"], 
        strategy="hi_res", 
        pdf_infer_table_structure="true",
        coordinates="true",
        extract_image_block_types=['Image', 'Table'],
        # Post processing to aggregate text once we have the title
        chunking_strategy = "by_title",
        combine_under_n_chars = 10,
        #include_orig_elements = 'true',
        max_characters = 10000,
        multipage_sections='True',
        new_after_n_chars = 10000,
        )
    return elements_by_title

def chunk_elements(elements, min_lenght=10, max_length=10000):
    """
    Chunk textual elements to the specified lenght. Images are excluded from chunking and handled separately

    Parameters
    ----------
    elements : list of Element Objects
        Contains the partition into the different info representations present in the PDF document.

    Returns
    -------
    list of Element Objects
        Contains the chunked elements and images contained in the partition.

    """
    image_elements = [el  for el in elements if "Image" in str(type(el))]
    textual_elements = [el  for el in elements if "Image" not in str(type(el))]
    chunked_by_title = chunk_by_title(elements=textual_elements,
                                      multipage_sections = True, # Default
                                      combine_text_under_n_chars = min_lenght, # Specifying 0 for this argument suppresses combining of small chunks
                                      new_after_n_chars = max_length, # Specifying 0 for this argument causes each element to appear in a chunk by itself (although an element with text longer than max_characters will be still be split into two or more chunks)
                                      max_characters = max_length # Cut off new sections after reaching a length of n chars (hard max). Default: 500
                                      )
    return image_elements + chunked_by_title


def categorize_elements(elements):
    """
    Postprocess Results to Element Objects
    
    Parameters
    ----------
    elements : list of Element Objects
        BaseModel-Object that stores the information retrieved by unstructured during PDF document partition.

    Returns
    -------
    table_elements : List[Element]
        Containing table elements.
    image_elements : List[Element]
        Containing image elements.
    text_elements : List[Element]
        Containing text elements.

    """
    
    categorized_elements = []
    for element in elements:
        
        if "Image" in str(type(element)):
            categorized_elements.append(Element(type="image", 
                                                element_id = element.id, # TODO: SHA-256 of element text is used --> set unique_element_ids=True in partitioning for uuid-garanteed uniqueness.
                                                text = element.text, 
                                                languages = element.metadata.languages, 
                                                page_number = element.metadata.page_number, 
                                                image_base64 = element.metadata.image_base64,
                                                image_mime_type = element.metadata.image_mime_type,
                                                filename = element.metadata.filename, 
                                                parent_id = element.metadata.parent_id, 
                                                text_as_html = element.metadata.text_as_html,
                                                #coordinates = element.metadata.coordinates.to_dict()  # requires update of unstructured for CompositeElements to contain orig_elements in metadata
                                                #relevance = element.relevance
                                                )
                                        )   
        
        # skip empty text and table elements and elements containing page numbers
        elif len(element.text) <= 3:
            continue
        
        elif "Table" in str(type(element)):
            categorized_elements.append(Element(type="table", 
                                                element_id = element.id, 
                                                text = element.text, 
                                                languages = element.metadata.languages, 
                                                page_number = element.metadata.page_number, 
                                                image_base64 = element.metadata.image_base64,
                                                image_mime_type = element.metadata.image_mime_type,
                                                filename = element.metadata.filename, 
                                                parent_id = element.metadata.parent_id, 
                                                text_as_html = element.metadata.text_as_html,
                                                #coordinates = element.metadata.coordinates.to_dict()  # requires update of unstructured for CompositeElements to contain orig_elements in metadata
                                                #relevance = element.relevance
                                                )
                                        )
        # TODO: prüfen, wie weitere Elementtypen alternativ abgelegt werden können
        else:
            categorized_elements.append(Element(type="text", 
                                                element_id = element.id, 
                                                text = element.text, 
                                                languages = element.metadata.languages, 
                                                page_number = element.metadata.page_number, 
                                                image_base64 = element.metadata.image_base64,
                                                image_mime_type = element.metadata.image_mime_type,
                                                filename = element.metadata.filename, 
                                                parent_id = element.metadata.parent_id, 
                                                text_as_html = element.metadata.text_as_html,
                                                #coordinates = element.metadata.coordinates.to_dict()   # requires update of unstructured for CompositeElements to contain orig_elements in metadata
                                                #relevance = element.relevance
                                                )
                                        )
            
    #  Split Elements by Element type
    #  Tables
    table_elements = [e for e in categorized_elements if e.type == "table"]
    #print("\n\nTables:", len(table_elements))
    
    #  Images
    image_elements = [e for e in categorized_elements if e.type == "image"]
    #print("\n\nImages:", len(image_elements))
    
    #  Text
    text_elements = [e for e in categorized_elements if e.type == "text"]
    #print("\n\nTexts:", len(text_elements))
    
    return table_elements, image_elements, text_elements

def categorize_elements_from_dict(elements):
    """
    Postprocess Results to Element Objects
    
    Parameters
    ----------
    elements : list of dictionaries containing Element Objects
        BaseModel-Object that stores the information retrieved by unstructured during PDF document partition.

    Returns
    -------
    table_elements : List[Element]
        Containing table elements.
    image_elements : List[Element]
        Containing image elements.
    text_elements : List[Element]
        Containing text elements.

    """
    
    categorized_elements = []
    for element in elements:
        
        if element['type']=="Image":
            categorized_elements.append(Element(type="image", 
                                                element_id = element['element_id'], # TODO: SHA-256 of element text is used --> set unique_element_ids=True in partitioning for uuid-garanteed uniqueness.
                                                text = element['text'], 
                                                languages = element['metadata']['languages'], 
                                                page_number = element['metadata']['page_number'], 
                                                image_base64 = element['metadata']['image_base64'],
                                                image_mime_type = element['metadata']['image_mime_type'],
                                                filename = element['metadata']['filename'],
#                                                parent_id = element['metadata']['parent_id'], 
#                                                text_as_html = element['metadata']['text_as_html'],
                                                #coordinates = element.metadata.coordinates.to_dict()  # requires update of unstructured for CompositeElements to contain orig_elements in metadata
                                                #relevance = element['relevance']
                                                )
                                        )   
        
        # skip empty text and table elements and elements containing page numbers
        elif len(element['text']) <= 3:
            continue
        
        elif element['type']=="Table":
            categorized_elements.append(Element(type="table", 
                                                element_id = element['element_id'], # TODO: SHA-256 of element text is used --> set unique_element_ids=True in partitioning for uuid-garanteed uniqueness.
                                                text = element['text'], 
                                                languages = element['metadata']['languages'], 
                                                page_number = element['metadata']['page_number'], 
                                                image_base64 = element['metadata']['image_base64'],
                                                image_mime_type = element['metadata']['image_mime_type'],
                                                filename = element['metadata']['filename'], 
#                                                parent_id = element['metadata']['parent_id'], 
                                                text_as_html = element['metadata']['text_as_html'],
                                                #coordinates = element.metadata.coordinates.to_dict()  # requires update of unstructured for CompositeElements to contain orig_elements in metadata
                                                #relevance = element['relevance']
                                                )
                                        )
        # TODO: prüfen, wie weitere Elementtypen alternativ abgelegt werden können
        else:
            categorized_elements.append(Element(type="text", 
                                                element_id = element['element_id'], # TODO: SHA-256 of element text is used --> set unique_element_ids=True in partitioning for uuid-garanteed uniqueness.
                                                text = element['text'], 
                                                languages = element['metadata']['languages'], 
                                                page_number = element['metadata']['page_number'], 
#                                                image_base64 = element['metadata']['image_base64'],
#                                                image_mime_type = element['metadata']['image_mime_type'],
                                                filename = element['metadata']['filename'], 
#                                                parent_id = element['metadata']['parent_id'], 
#                                                text_as_html = element['metadata']['text_as_html'],
                                                #coordinates = element.metadata.coordinates.to_dict()  # requires update of unstructured for CompositeElements to contain orig_elements in metadata
                                                #relevance = element['relevance']
                                                )
                                        )
            
    #  Split Elements by Element type
    #  Tables
    table_elements = [e for e in categorized_elements if e.type == "table"]
    #print("\n\nTables:", len(table_elements))
    
    #  Images
    image_elements = [e for e in categorized_elements if e.type == "image"]
    #print("\n\nImages:", len(image_elements))
    
    #  Text
    text_elements = [e for e in categorized_elements if e.type == "text"]
    #print("\n\nTexts:", len(text_elements))
    
    return table_elements, image_elements, text_elements

@traceable(
    task="text/table summarization",
    version="base"
    )
@backoff.on_exception(backoff.expo, RateLimitError)
def summarize_elements(elements, model):
    """
    Summarize chain for elements containing text.

    Parameters
    ----------
    elements : List of strings
        Contains texts to be summarized.
    model : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for text summarization.

    Returns
    -------
    List of strings
        Contains summaries of element texts.

    """
    prompt_text = """
                    You are an assistant tasked with summarizing tables and text. \
                    Give a concise summary of the table or text. Table or text chunk: {element} 
                """
                    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    
    summaries = summarize_chain.batch(elements, {"max_concurrency": 3})

    return summaries

@traceable(
    task="text/table summarization",
    version="base"
    )
@backoff.on_exception(backoff.expo, RateLimitError)
def summarize_text(elements, model):
    """
    Summarize chain for elements containing text.

    Parameters
    ----------
    elements : List of strings
        Contains texts to be summarized.
    model : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for text summarization.

    Returns
    -------
    List of strings
        Contains summaries of element texts.

    """
    prompt_text = """
                    You are an assistant tasked with summarizing text from climate policy documents. \
                    Give a concise summary of the text chunk: {element} 
                """
                    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    
    summaries = summarize_chain.batch(elements, {"max_concurrency": 3})

    return summaries

@traceable(
    task="text/table summarization",
    version="base"
    )
@backoff.on_exception(backoff.expo, RateLimitError)
def summarize_table(elements, model):
    """
    Summarize chain for elements containing tables.

    Parameters
    ----------
    elements : List of strings
        Contains texts to be summarized.
    model : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for text summarization.

    Returns
    -------
    List of strings
        Contains summaries of element texts.

    """
    prompt_text = """
                    You are an assistant tasked with summarizing tables from climate policy documents. \
                    Give a concise summary of the table: {element} 
                """
                    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    
    summaries = summarize_chain.batch(elements, {"max_concurrency": 3})

    return summaries

def encode_image(image_path):
    """
    This function takes the path to an image file as input and returns the Base64 encoded string representation of the image.
    --> https://base64.guru/converter/decode/image

    Parameters
    ----------
    image_path : String
        The path to the image file that needs to be encoded.

    Returns
    -------
    String
        The Base64 encoded string representation of the image.

    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def retrieve_images_from_folder(image_folder):
    """
    Retrieve images from folder and encode them to Base64.

    Parameters
    ----------
    image_folder : str
        Folder path, containing the images.

    Returns
    -------
    encoded_images : list[str]
        List of base64 encodings, each one representing one image from the folder.

    """
    images = []
    encoded_images = []
    
    #  List files in img directory
    image_files = os.listdir(image_folder)
    
    for file in image_files:
        # make sure file is an image
        if file.endswith(('.jpg', '.png', 'jpeg')):
            img_path = image_folder + file
            images.append(img_path)
            
            encoded_images.append(encode_image(img_path))
            
    return encoded_images

@traceable(
    task="image summarization",
    version="base"
    )
def summarize_image(encoded_image, model):
    """
    Function for image summaries.
    
    Adapted from
    https://github.com/Coding-Crashkurse/Multimodal-RAG-With-OpenAI/blob/main/Semi_structured_and_multi_modal_RAG%20(1).ipynb

    Parameters
    ----------
    encoded_image : str
        Base64-encoded image.

    Returns
    -------
    String
        Summary of image.

    """
    prompt = [   
        AIMessage(
            content="You are a useful bot that is especially good at OCR from images"
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": """Given the image, provide the following information:
                                            - A detailed description of the image. Be specific about graphs, such as bar plots.
                                            - Classify which kind of information is contained in the image (picture, graph, table, etc.)
                                            - The text contained in the image. If necessary, provide a concise summary."""},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    },
                },
            ]
        )
    ]    
    response = model.invoke(prompt)
    return response.content

def add_documents_to_vectorstore(vectorstore, summaries, original_contents):
    """
    Function to add documents to the vectorstore. 
    If/else are intrdocued for table documents containing html representation and elements not containing parent elements.

    Parameters
    ----------
    vectorstore : langchain_pinecone.vectorstores.PineconeVectorStore
    
    summaries : list of strings
        
    original_contents : Element(BaseModel)

    Returns
    -------
    None.

    """
    # Doc ID only needed for retriever with in-memory docstore 
    #doc_ids = [str(uuid.uuid4()) for _ in summaries]
    
    # https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html
    summary_docs = [
        #no html representation and no parent element:
        Document(page_content=s, metadata={#id_key : doc_ids[i],
                                           'element_id' : original_contents[i].element_id, 
                                           'type' : original_contents[i].type,
                                           'original_content' : original_contents[i].text, 
                                           #TODO: 'languages' : original_contents[i].languages, 
                                           'page_number' : original_contents[i].page_number, 
                                           #TODO: 'image_base64' : original_contents[i].image_base64,
                                           #TODO: 'image_mime_type' : original_contents[i].image_mime_type,
                                           'filename' : original_contents[i].filename, 
                                           'parent_id' : 'None', 
                                           'text_as_html' : 'None',
                                           #'coordinates' : str(original_contents[i].coordinates['points'])
                                           #'relevance' : original_contents[i].relevance
                                           }
                 )
        if original_contents[i].text_as_html == None and original_contents[i].parent_id == None
        else
        #no html representation
        Document(page_content=s, metadata={#id_key : doc_ids[i],
                                          'element_id' : original_contents[i].element_id, 
                                          'type' : original_contents[i].type,
                                          'original_content' : original_contents[i].text, 
                                          #TODO: 'languages' : original_contents[i].languages, 
                                          'page_number' : original_contents[i].page_number, 
                                          #TODO: 'image_base64' : original_contents[i].image_base64,
                                          #TODO: 'image_mime_type' : original_contents[i].image_mime_type,
                                          'filename' : original_contents[i].filename, 
                                          'parent_id' : original_contents[i].parent_id, 
                                          'text_as_html' : 'None',
                                          #'coordinates' : str(original_contents[i].coordinates['points'])
                                          #'relevance' : original_contents[i].relevance
                                          }
                )
        if original_contents[i].text_as_html == None and original_contents[i].parent_id != None
        else
        #no parent_id representation
        Document(page_content=s, metadata={#id_key : doc_ids[i],
                                          'element_id' : original_contents[i].element_id, 
                                          'type' : original_contents[i].type,
                                          'original_content' : original_contents[i].text, 
                                          #TODO: 'languages' : original_contents[i].languages, 
                                          'page_number' : original_contents[i].page_number, 
                                          #TODO: 'image_base64' : original_contents[i].image_base64,
                                          #TODO: 'image_mime_type' : original_contents[i].image_mime_type,
                                          'filename' : original_contents[i].filename, 
                                          'parent_id' : 'None', 
                                          'text_as_html' : original_contents[i].text_as_html,
                                          #'coordinates' : str(original_contents[i].coordinates['points'])
                                          #'relevance' : original_contents[i].relevance
                                          }
                )
        if original_contents[i].text_as_html != None and original_contents[i].parent_id == None
        else
        Document(page_content=s, metadata={#id_key : doc_ids[i],
                                           'element_id' : original_contents[i].element_id, 
                                           'type' : original_contents[i].type,
                                           'original_content' : original_contents[i].text, 
                                           #TODO: 'languages' : original_contents[i].languages, 
                                           'page_number' : original_contents[i].page_number, 
                                           #TODO: 'image_base64' : original_contents[i].image_base64,
                                           #TODO: 'image_mime_type' : original_contents[i].image_mime_type,
                                           'filename' : original_contents[i].filename, 
                                           'parent_id' : original_contents[i].parent_id, 
                                           'text_as_html' : original_contents[i].text_as_html,
                                           #'coordinates' : str(original_contents[i].coordinates['points'])
                                           #'relevance' : original_contents[i].relevance
                                           }
                 )
       for i, s in enumerate(summaries)
    ]
    vectorstore.add_documents(summary_docs)