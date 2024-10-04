# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:36:39 2024

@author: becker_nic

QUOTE EXTRACTION MODULE
"""

import pprint
import copy

from tqdm import tqdm
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import JsonOutputParser # "Probably the most reliable output parser for getting structured data that does NOT use function calling." (https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/ )
from langchain_core.output_parsers import PydanticOutputParser # https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/pydantic/
from langchain_core.prompts import PromptTemplate
from langsmith import traceable
from general_utils import setup_logger

# Set logging
logger = setup_logger(__name__, 'logfile.log')

class QuoteObject(BaseModel):
    quotes: Optional[List[str]]
    
def get_quote_extraction_JSON_chain(llm):
    """
    Creates a chain able to extract quotes from a provided document snippet.

    Parameters
    ----------
    llm : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for quote extraction.

    Returns
    -------
    quote_extraction_chain : langchain_core.runnables.base.RunnableSequence
        Chain object providing the prompt and output parser for quote extraction.

    """
    # Set up a parser + inject instructions into the prompt template.
    json_parser = JsonOutputParser(pydantic_object=QuoteObject)

    quote_extraction_prompt = PromptTemplate(
                                            template="You are a climate and transport policy analyst, specialized in Nationally Determined Contributions (NDCs). You will be provided with climate policy document snippets from NDCs. Your task is to extract quotes from these sippets. \
                                                    The quotes should define targets, measures and or actions undertaken or planned.\
                                                    If the snippet does not contain relevant information, return an empty answer. Formulate your response ONLY based on the information contained in the document snippet. \
                                                    BEFORE responding, CHECK whether the quote reproduces the EXACT wording from the document.\
                                                    Answer providing a JSON structure, where each key-value pair represents one quote.\
                                                    \nHere is the NDC snippet:{Context}\n{format_instructions}",
                                            input_variables=["Context"],
                                            partial_variables={"format_instructions": json_parser.get_format_instructions()},
                                            )

    quote_extraction_chain = quote_extraction_prompt | llm | json_parser
    
    return quote_extraction_chain

def get_quote_extraction_Pydantic_chain(llm):
    """
    Creates a chain able to extract quotes from a provided document snippet.

    Parameters
    ----------
    llm : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for quote extraction.

    Returns
    -------
    quote_extraction_chain : langchain_core.runnables.base.RunnableSequence
        Chain object providing the prompt and output parser for quote extraction.

    """
    # Set up a parser + inject instructions into the prompt template.
    pydantic_parser = PydanticOutputParser(pydantic_object=QuoteObject)

    quote_extraction_prompt = PromptTemplate(
                                            template="You are a climate and transport policy analyst, specialized in Nationally Determined Contributions (NDCs). You will be provided with climate policy document snippets from NDCs. Your task is to extract quotes from these sippets. \
                                                    The quotes should define targets, measures and or actions undertaken or planned.\
                                                    If the snippet does not contain relevant information, return an empty answer. Formulate your response ONLY based on the information contained in the document snippet. \
                                                    BEFORE responding, CHECK whether the quote reproduces the EXACT wording from the document.\
                                                    \nHere is the NDC snippet:{Context}\n{format_instructions}",
                                            input_variables=["Context"],
                                            partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
                                            )

    quote_extraction_chain = quote_extraction_prompt | llm | pydantic_parser
    
    return quote_extraction_chain


@traceable(
    task="quote extraction",
    version="revision"
    )
def quotes_revision_chain(llm):
    """
    Revise the relevance of extracted quotes.

    Parameters
    ----------
    llm : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for quote revision.

    Returns
    -------
    quote_revision_chain : langchain_core.runnables.base.RunnableSequence
        Chain object providing the prompt and output parser for quote revision.

    """
    # Set up a parser + inject instructions into the prompt template.
    pydantic_parser = PydanticOutputParser(pydantic_object=QuoteObject)

    quote_revision_prompt = PromptTemplate(
                                            template="You are a climate and transport policy analyst, specialized in Nationally Determined Contributions (NDCs). "                                                  
                                                    "You will be provided with a list of quotes from a climate policy document, like an NDCs. \
                                                    Your task is to select those quotes, that define targets, measures and or actions undertaken or planned.\
                                                    The selected targets, measures or actions must be either from the transport or the energy sector, or define economy-wide greenhouse gas (GHG) reduction targets.\
                                                    Do NOT alter any of the quotes.\
                                                    In case the input list is empty, respond with an empty answer as well. \
                                                    \nHere is the NDC snippet:{Quotes}\n{format_instructions}",
                                            input_variables=["Quotes"],
                                            partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
                                            )

    quote_revision_chain = quote_revision_prompt | llm | pydantic_parser
    
    return quote_revision_chain


@traceable(
    task="quote extraction",
    version="base"
    )
def get_quotes_with_keywords(llm, doc_dict):
    """
    Function extracting quotes from the retrieved documents. Returns a dictionary object containing the quotes along with relevant metadata.

    Parameters
    ----------
    llm : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for quote extraction.
    doc_dict : Dict
        Dictionary containing the retrieved documents from vectorstore, including the original contents.

    Returns
    -------
    quotes_dict : Dict
        Dictionary containing the quotes corresponding to each retrieved document.

    """
    
    #initialize result dictionary
    quotes_dict = {} 
    memory = []

    for keyword, key_dict in tqdm(doc_dict.items()):
        for key, value in key_dict.items():
            #check for duplicates in retrieved docs
            #if document was already analyzed, append keyword to keyword list
            if value['unstr_element_id'] in memory:
                for k,v in quotes_dict.items():
                    if value['unstr_element_id'] == v['unstr_element_id']:
                        pprint.pprint("\nError:")
                        pprint.pprint(v)
                        v['keywords'].append(keyword)
                continue
            else:
                memory.append(value['unstr_element_id'])
            
                if value['type'] == 'image':
                    snippet = value['summary']
                
                # elif value['type'] == 'table':
                    #snippet = value['text_as_html'] # --> WORSE results observed from using html for quote extraction
                else:
                    snippet = value['content']
               
                # invoke quote extraction chain
                quote_extraction_chain = get_quote_extraction_Pydantic_chain(llm)
                try:
                    response = quote_extraction_chain.invoke({"Context": snippet})
                except Exception as e:
                    print(f"Exception {e} occured during quote extraction.")
                    logger.exception(f'Exception {e} occured during quote extraction')
                    
                # add quotes to general results
                i = len(quotes_dict)
                quotes_dict[i] = copy.deepcopy(value)
                quotes_dict[i].pop('similarity_score', None)
                quotes_dict[i]['keywords'] = [keyword]
                try:
                    quotes_dict[i]['quotes'] = []
                    quotes_dict[i]['unverified_quotes'] = []
                    # quote verification - check if quote is in snippet --> avoid HALUSCINATION
#                    for q in response['quotes']: # for JSON chain --> replaces by Pydantic chain as it is (50%) faster 
                    for q in response.quotes: # for dict
                        index = snippet.find(q)
                        if index != -1: # q found in snippet
                            quotes_dict[i]['quotes'].append((index, q))
                        else: # q NOT found in snippet
                            quotes_dict[i]['unverified_quotes'].append((index, q))
                            
                except KeyError: 
                    # no quotes in provided content
                    quotes_dict[i]['quotes'] = []
                    quotes_dict[i]['unverified_quotes'] = []
                except Exception as e:
                    logger.exception(f'Exception {e} while processing quotes from extraction chain.')
                    
                #pprint.pprint(quotes_dict[i])
                
    return quotes_dict


@traceable(
    task="quote extraction",
    version="base"
    )
def get_quotes(llm, doc_dict):
    """
    Function extracting quotes from the retrieved documents. Returns a dictionary object containing the quotes along with relevant metadata.

    Parameters
    ----------
    llm : langchain_openai.chat_models.base.ChatOpenAI
        Language Model to be used for quote extraction.
    doc_dict : Dict
        Dictionary containing the retrieved documents from vectorstore, including the original contents.

    Returns
    -------
    quotes_dict : Dict
        Dictionary containing the quotes corresponding to each retrieved document.

    """
    
    #initialize result dictionary
    quotes_dict = {} 

    for key, value in tqdm(doc_dict.items()):
                    
        if value['type'] == 'image':
            snippet = value['summary']
        
        else:
            snippet = value['content']
           
        # invoke quote extraction chain
        quote_extraction_chain = get_quote_extraction_Pydantic_chain(llm)
        
        try:
            response_pydantic = quote_extraction_chain.invoke({"Context": snippet})
            response = quotes_revision_chain(llm).invoke({"Quotes": response_pydantic.quotes})
        except Exception as e:
            print(f"Exception {e} occured during quote extraction.")
            logger.exception(f'Exception {e} occured during quote extraction')
            
        # add quotes to general results
        i = len(quotes_dict)
        quotes_dict[i] = copy.deepcopy(value)
        quotes_dict[i].pop('similarity_score', None)
        
        try:
            quotes_dict[i]['quotes'] = []
            quotes_dict[i]['unverified_quotes'] = []
            # quote verification - check if quote is in snippet --> avoid HALUSCINATION
            for q in response.quotes: # for dict --> replaced JSON with Pydantic chain as it is (50%) faster 
                index = snippet.find(q)
                if index != -1: # q found in snippet
                    quotes_dict[i]['quotes'].append((index, q))
                else: # q NOT found in snippet
                    quotes_dict[i]['unverified_quotes'].append((index, q))
                    
        except KeyError: 
            # no quotes in provided content
            quotes_dict[i]['quotes'] = []
            quotes_dict[i]['unverified_quotes'] = []
        except Exception as e:
            logger.exception(f'Exception {e} while processing quotes from extraction chain.')
            
                
    return quotes_dict


