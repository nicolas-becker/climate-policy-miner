# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:57:48 2024

@author: becker_nic

General Utils
"""
import os
import re
import fitz #PyMuPDF
import requests
import urllib.parse
import pandas as pd
import logging
import unicodedata
import numpy as np

#from fuzzywuzzy import fuzz
from tqdm import tqdm


def compare_quotes_fuzzy(ground_truth, matched, threshold=80):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    matched_set = set(matched)
    ground_truth_set = set(ground_truth)
    
    for gt_quote in ground_truth_set:
        max_similarity = 0
        for matched_quote in matched_set:
            similarity = fuzz.ratio(gt_quote, matched_quote)
            if similarity > max_similarity:
                max_similarity = similarity
        
        if max_similarity >= threshold:
            true_positives += 1
        else:
            false_negatives += 1
    
    for matched_quote in matched_set:
        max_similarity = 0
        for gt_quote in ground_truth_set:
            similarity = fuzz.ratio(matched_quote, gt_quote)
            if similarity > max_similarity:
                max_similarity = similarity
        
        if max_similarity < threshold:
            false_positives += 1
    
    return true_positives, false_positives, false_negatives

def namespace_exists(index, namespace):
    """
    Check if namespace exists in Pinecone index.
    From: https://community.pinecone.io/t/how-do-i-check-if-a-namespace-already-exists-in-an-index/2684/6

    Parameters
    ----------
    index : TYPE
        DESCRIPTION.
    namespace : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    namespaces = index.describe_index_stats()['namespaces']
    namespace_in_index = namespace in namespaces
    return namespace_in_index

def create_highlighted_pdf(filename, quotes, output_path, df=None, color='yellow', opacity=0.5):
    #TODO - create link to quote in document --> https://pymupdf.readthedocs.io/en/latest/the-basics.html#getting-page-links
    """
    Create highlighted PDF with PyMuPDF library. All extracted quotes are highlighted.
    https://pymupdf.readthedocs.io/en/latest/index.html
    
    Parameters
    ----------
    filename : str
        Path to PDF that is analyzed.
    quotes : iterable
        List or pd.Series object containing the quotes to be highlighted in the text.
    output_path : str
        Destination path for the highlighted document.
    df : pd.DataFrame
        DataFrame containing additional information to the quotes

    Returns
    -------
    quote_links : list(str)
        List containing links to the quotes in the document
        
    """
    #TODO: Highlighting Scanned Text - OCR
    
    # read pdf with pymudpdf
    doc = fitz.open(filename)
    #quote_links = []
    for i, quote in tqdm(enumerate(quotes)): 
        if df is not None:
            row=df.iloc[i]
        else:
            # of no df is provided, all quotes are highlighted as targets
            row={}
            row['target'] = 'True'
            row['measure'] = 'False'
            
        for page_number in range(len(doc)):
            page = doc[page_number] 
            # search for quote on page
            text_instances = page.search_for(quote)
            
            # highlight
            for inst in text_instances:
                if row['target'] == 'True': # targets highlighted in yellow
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_opacity(opacity)  # Set opacity (0.0 to 1.0)
                    highlight.update() 
                
                if row['measure'] == 'True': # measures highlighted in blue
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=(0, 1, 1))  # RGB for cyan
                    highlight.set_opacity(opacity)  # Set opacity (0.0 to 1.0)
                    highlight.update()
                    
                # annotation with comments containing classification
                #if (quote.target_labels != 'None') | (quote.measure_labels != 'None'):
                #    highlight.set_info(content=f"Target Labels: {quote.target_labels}; Measure Labels: {quote.measure_labels}")
                
                # Create a link annotation for the highlighted text
                #link = page.add_link_annot(inst, page_number, -1)
                #quote_links.append(link)
            
    # save output --> https://pymupdf.readthedocs.io/en/latest/document.html#Document.save
    doc.save(output_path, garbage=4, deflate=True, clean=True)
    
    #return quote_links


def setup_logger(name, log_file, level=logging.DEBUG):
    """
    Function to setup a logger.
    
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Set logging
logger = setup_logger(__name__, 'logfile.log')


def is_valid_url(url):
    """
    Check if a URL is valid using a regular expression.
    
    Parameters:
    url (str): The URL to validate.
    
    Returns:
    bool: True if the URL is valid, False otherwise.
    """
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, str(url)) is not None


def download_pdf_wget(url, output_path):
    """
    Download a PDF file from the given URL using wget.

    Parameters:
    url (str): The URL of the PDF file.
    output_path (str): The path where the downloaded PDF will be saved.
    """
    wget.download(url, out=output_path)
    


def download_pdf_urllib(url, output_path):
    """
    Download a PDF file from the given URL using urllib.

    Parameters:
    url (str): The URL of the PDF file.
    output_path (str): The path where the downloaded PDF will be saved.
    """
    urllib.request.urlretrieve(url, output_path)    


def extract_text_from_pdf(pdf_path):
    """
    Extract the text from the PDF document.

    Parameters
    ----------
    pdf_path : TYPE
        DESCRIPTION.

    Returns
    -------
    text : TYPE
        DESCRIPTION.

    """
    document = fitz.open(pdf_path)
    text = ''
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text


def find_text_context(text, search_text, context_length=400):
    """
    Search for the specified text within the extracted text.
    Extract the preceding 100 characters and following 100 characters around the found text.
    Save the extracted information back to your dataset.

    Parameters
    ----------
    text : TYPE
        DESCRIPTION.
    search_text : TYPE
        DESCRIPTION.
    context_length : TYPE, optional
        DESCRIPTION. The default is 400. --> extacts text chunks of around 1000 chars - hard test conditions. Alternative: context_length=200 for easier conditions.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    start_idx = text.find(search_text)
    if start_idx == -1:
        return None, None
    start_context = max(start_idx - context_length, 0)
    end_context = min(start_idx + len(search_text) + context_length, len(text))
    context = text[start_context:end_context]
    return start_context, context



def download_pdf(url, output_path):
    """
    Download the PDF document from the given URL using requests.

    Parameters
    ----------
    url : TYPE
        DESCRIPTION.
    output_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    response = requests.get(url)
    with open(output_path, 'wb') as file:
        file.write(response.content)
        
        
def decode_filenames_in_directory(directory):
    """
    Decode all filenames in the specified directory that contain URL-encoded characters.

    Parameters:
    directory (str): The path to the directory containing the files to be renamed.
    """
    for filename in os.listdir(directory):
        decoded_filename = urllib.parse.unquote(filename)
        if filename != decoded_filename:
            original_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, decoded_filename)
            os.rename(original_file_path, new_file_path)
            print(f'Renamed: {filename} -> {decoded_filename}')


def check_string_in_list(row: pd.Series, true:str, choice:str) -> str:
    """
    Function to check if the string 'true' is in the list 'choice'.

    Parameters
    ----------
    row : pd.Series
        DESCRIPTION.
    true : str
        DESCRIPTION.
    choice : str
        DESCRIPTION.

    Returns
    -------
    str
        DESCRIPTION.

    """
    return row[true] if row[true] in row[choice] else np.nan


def normalize_filename(filename):
    """
    Normalize the filename to remove non-ASCII characters
    """
    normalized_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
    return normalized_filename


def plot_multiclass_precision_recall(y_score, y_true_untransformed, class_list, classifier_name):
    """
    Precision-Recall plotting for a multiclass problem. It plots average precision-recall, per class precision recall and reference f1 contours.

    Code slightly modified, but heavily based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    
    From: https://github.com/openai/openai-cookbook/blob/main/examples/utils/embeddings_utils.py
    """
    n_classes = len(class_list)
    y_true = pd.concat(
        [(y_true_untransformed == class_list[i]) for i in range(n_classes)], axis=1
    ).values

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision_micro, recall_micro, _ = precision_recall_curve(
        y_true.ravel(), y_score.ravel()
    )
    average_precision_micro = average_precision_score(y_true, y_score, average="micro")
    print(
        str(classifier_name)
        + " - Average precision score over all classes: {0:0.2f}".format(
            average_precision_micro
        )
    )

    # setup plot details
    plt.figure(figsize=(9, 10))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append("iso-f1 curves")
    (l,) = plt.plot(recall_micro, precision_micro, color="gold", lw=2)
    lines.append(l)
    labels.append(
        "average Precision-recall (auprc = {0:0.2f})" "".format(average_precision_micro)
    )

    for i in range(n_classes):
        (l,) = plt.plot(recall[i], precision[i], lw=2)
        lines.append(l)
        labels.append(
            "Precision-recall for class `{0}` (auprc = {1:0.2f})"
            "".format(class_list[i], average_precision[i])
        )

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{classifier_name}: Precision-Recall curve for each class")
    plt.legend(lines, labels)
