# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:31:24 2024

@author: becker_nic

DOC RETRIEVAL MODULE
"""
import fitz
import ast
from tqdm import tqdm
from general_utils import setup_logger

# Set logging
logger = setup_logger(__name__, 'logfile.log')



def pixels_to_points(pixel_bbox, dpi=150):
    points_per_inch = 72  # 1 point = 1/72 inch
    scale = points_per_inch / dpi
    return [coord * scale for coord in pixel_bbox]


def highlight_retrieved_elements(filename, output_path, elements):
    # Open the PDF document
    doc = fitz.open(filename)
    
    for element in elements:
            
        page = doc[int(element['page_number'])-1]
        
        coordinates = ast.literal_eval(element['coordinates'])
        
        rect_pixels = tuple((coordinates[0][0], coordinates[0][1], coordinates[2][0], coordinates[2][1])) # Bounding Box: (x0, y0, x1, y1) - “top left” corner P(x0,y0) and P(x1,y1) as the “bottom right” one
        rect = pixels_to_points(rect_pixels)
           
        # Create a rectangle for the highlight area
        highlight_rect = fitz.Rect(rect) # Rect represents a rectangle defined by four floating point numbers x0, y0, x1, y1. They are treated as being coordinates of two diagonally opposite points. The first two numbers are regarded as the “top left” corner P(x0,y0) and P(x1,y1) as the “bottom right” one.
        
        # Add the highlight annotation
        highlight = page.add_highlight_annot(highlight_rect)
        highlight.set_opacity(0.5)  # Optional: Set opacity
        highlight.update()
    
    # Save the modified PDF
    doc.save(output_path, garbage=4, deflate=True, clean=True)


### DEPRECATED
def filter_by_score(d, threshold):
    """
    DEPRECATED
    
    Set similarity threshold d for query

    Parameters
    ----------
    d : float
        similarity threshold.

    Returns
    -------
    Str
        Filter Option for Pinecone index.query()

    """
    return d["score"] > threshold

def get_docs_from_vectorstore(vectorstore, index_name, namespace, query, embedding, k, score_threshold=0.5):
    """
    Variation of:  https://docs.pinecone.io/guides/data/query-data
    --> https://sdk.pinecone.io/typescript/types/QueryResponse.html
    and LangChain_Pinecone.similarity_search_by_vector_wtih_score()
    --> https://api.python.langchain.com/en/latest/_modules/langchain_pinecone/vectorstores.html#PineconeVectorStore.similarity_search
    
    Parameters
    ----------
    vectorstore : langchain_pinecone.vectorstores.PineconeVectorStore
    
    index_name : str
        Name of the Index to be retrieved in Pinecone VectorStore 
    
    namespace : str
        Name of the Namespace to be retrieved in Index 
        
    query : string
        Keyword or similar query to be applied for semantic search.
        
    k : number of results
    
    score_threshold : similarity threshold

    Returns
    -------
    docs : Dict
        Contains results of query with:
            - similarity_score : Similarity score from search (Cosine Similarity)
            - vs_id : Document IDs by VectorStore
            - content : Original text contained in retrieved element
            - metadata : Metadata included in VectorStore (https://sdk.pinecone.io/typescript/types/PineconeRecord.html)
                            - "type" 
                            - "text_as_html"
                            - "unstr_element_id"
                            - "unstr_parent_id" 
                            - "filename"
                            - "page_number" 
        for each result.
    """
    print(f'searching pinecone index "{index_name}" in namespace "{namespace}" \nwith query: "{query}"')
    
    index = vectorstore.get_pinecone_index(index_name)
    input_vector = embedding.embed_query(query)
    results = index.query(namespace=namespace, vector=input_vector, top_k=k, include_metadata=True, include_values=True)
    
    #extract tuples
    docs = []
    for result in results['matches']:
        if result["score"] >= score_threshold:
            id_ = result['id'],
            vector = result['values'],
            metadata = result["metadata"],
            score = result["score"],
            docs.append((id_[0], vector[0], metadata[0], score[0]))
      
    #format as dictionary
    rel_docs = {}
    for i, doc in enumerate(docs):
        rel_docs[i] = {
            "similarity_score" : doc[3],
            "vs_id" : doc[0],
            "type" : doc[2]['type'],
            "content" : doc[2]['original_content'], 
            "summary" : doc[2]['text'], # needed for immages
            "text_as_html" : doc[2]['text_as_html'],
            "unstr_element_id" : doc[2]['element_id'],
            "unstr_parent_id" : doc[2]['parent_id'],
            "filename" : doc[2]['filename'],
            "page_number" : doc[2]['page_number'],
            #"coordinates" : doc[2]['coordinates'],
            #"relevance" : doc[2]['relevance']
            }
    
    return rel_docs

