# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:57:48 2024

@author: becker_nic

General Utils
"""
import os
import re
import math
import fitz #PyMuPDF
import requests
import urllib.parse
import pandas as pd
import logging
import unicodedata
import numpy as np
import json

def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None so JSON is valid."""
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        # Replace NaN/Inf with None before converting to dict
        df = obj.replace([np.nan, np.inf, -np.inf], None)
        return _sanitize_for_json(df.to_dict('records'))
    if isinstance(obj, pd.Series):
        s = obj.replace([np.nan, np.inf, -np.inf], None)
        return _sanitize_for_json(s.tolist())
    return obj

# Load configuration
def load_config(config_file):
    try:
        with open(config_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Config file '{config_file}' is not a valid JSON.")
        return {}

def compare_quotes_fuzzy(ground_truth, matched, threshold=80):
    """
    Compares two lists of quotes using fuzzy string matching and calculates
    the number of true positives, false positives, and false negatives.

    Parameters
    ----------
    ground_truth : list of str
        The list of ground truth quotes.
    matched : list of str
        The list of matched quotes to compare against the ground truth.
    threshold : int, optional
        The similarity threshold for considering a match. The default is 80.

    Returns
    -------
    tuple
        A tuple containing three integers:
            - true_positives (int): The number of quotes in `ground_truth` that have a match in `matched` with similarity >= `threshold`.
            - false_positives (int): The number of quotes in `matched` that do not have a match in `ground_truth` with similarity >= `threshold`.
            - false_negatives (int): The number of quotes in `ground_truth` that do not have a match in `matched` with similarity >= `threshold`.
    """
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
    Check if a namespace exists in a Pinecone index.

    This function queries the Pinecone index to retrieve the list of existing namespaces
    and checks if the specified namespace is present in that list.

    Parameters
    ----------
    index : pinecone.Index
        The Pinecone index object to query.
    namespace : str
        The namespace to check for existence in the index.
    
    Returns
    -------
    bool
        True if the namespace exists in the index, False otherwise.

    References
    .. [1] https://community.pinecone.io/t/how-do-i-check-if-a-namespace-already-exists-in-an-index/2684/6
    
    """
    namespaces = index.describe_index_stats()['namespaces']
    namespace_in_index = namespace in namespaces
    return namespace_in_index

def create_highlighted_pdf(filename, quotes, output_path, df=None, color='yellow', opacity=0.5):
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
    color : str
        Default highlight color (legacy parameter)
    opacity : float
        Opacity of highlights (0.0 to 1.0)

    Returns
    -------
    None
    """
    #TODO: Highlighting Scanned Text - OCR
    #TODO - create link to quote in document --> https://pymupdf.readthedocs.io/en/latest/the-basics.html#getting-page-links
    
    # Open PDF document
    doc = fitz.open(filename)
    
    # Add legend to the first page
    if len(doc) > 0:
        first_page = doc[0]
        add_legend_box(first_page)
    
    # Process quotes and highlight them
    for i, quote in enumerate(quotes):  # remove tqdm to avoid threading issues
        if df is not None:
            row=df.iloc[i]
        else:
            # If no df is provided, all quotes are highlighted as targets
            row = {}
            row['target'] = 'True'
            row['measure'] = 'False'
        
        # Determine highlight colors based on classification
        is_target = row.get('target', 'False') == 'True'
        is_measure = row.get('measure', 'False') == 'True'
        
        for page_number in range(len(doc)):
            page = doc[page_number] 
            # Search for quote on page
            text_instances = page.search_for(quote)
            
            # Highlight based on classification
            for inst in text_instances:
                if is_target and is_measure:
                    # Both target and measure - GREEN
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=(0, 1, 0))  # RGB for green
                    highlight.set_opacity(opacity)
                    highlight.update()
                    
                elif is_target:
                    # Target only - YELLOW (default)
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_opacity(opacity)
                    highlight.update()
                    
                elif is_measure:
                    # Measure only - BLUE/CYAN
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=(0, 1, 1))  # RGB for cyan
                    highlight.set_opacity(opacity)  # Set opacity (0.0 to 1.0)
                    highlight.update()
                    
                # annotation with comments containing classification
                #if (quote.target_labels != 'None') | (quote.measure_labels != 'None'):
                #    highlight.set_info(content=f"Target Labels: {quote.target_labels}; Measure Labels: {quote.measure_labels}")
            
                # TODO: Create a link annotation for the highlighted text
                #link = page.add_link_annot(inst, page_number, -1)
                #quote_links.append(link)
            
    # save output --> https://pymupdf.readthedocs.io/en/latest/document.html#Document.save
    doc.save(output_path, garbage=4, deflate=True, clean=True)
    doc.close()

def add_legend_box(page):
    """
    Add a legend box to the top-right corner of a PDF page.
    
    Parameters
    ----------
    page : fitz.Page
        The PDF page to add the legend to.
    """
    
    try:
        # Get page dimensions
        page_rect = page.rect
        
        # Define legend box position (top-right corner)
        legend_width = 150
        legend_height = 120
        margin = 20
        
        legend_rect = fitz.Rect(
            page_rect.width - legend_width - margin,
            margin,
            page_rect.width - margin,
            margin + legend_height
        )
        
        # Draw legend background
        page.draw_rect(legend_rect, color=(0, 0, 0), fill=(1, 1, 1), width=1)
        
        # Add legend title - USE BUILT-IN FONT
        title_point = fitz.Point(legend_rect.x0 + 10, legend_rect.y0 + 20)
        page.insert_text(
            title_point, 
            "Legend", 
            fontsize=12, 
            color=(0, 0, 0), 
            fontname="Times-Bold"  # ← CHANGE FROM "helv-bold" TO "Times-Bold"
        )
        
        # Add legend items
        legend_items = [
            ("Target", (1, 1, 0)),      # Yellow (default highlight color)
            ("Measure", (0, 1, 1)),     # Blue/Cyan
            ("Both", (0, 1, 0))         # Green
        ]
        
        y_offset = 35
        for item_text, color in legend_items:
            # Draw color box
            color_rect = fitz.Rect(
                legend_rect.x0 + 10,
                legend_rect.y0 + y_offset,
                legend_rect.x0 + 25,
                legend_rect.y0 + y_offset + 10
            )
            page.draw_rect(color_rect, fill=color, width=0)
            
            # Add text - USE BUILT-IN FONT
            text_point = fitz.Point(legend_rect.x0 + 30, legend_rect.y0 + y_offset + 8)
            page.insert_text(
                text_point, 
                item_text, 
                fontsize=10, 
                color=(0, 0, 0), 
                fontname="Times-Roman"  # ← CHANGE FROM "helv" TO "Times-Roman"
            )
            
            y_offset += 20
            
    except Exception as e:
        print(f"Warning: Could not add legend box: {e}")
        # Don't fail the entire process for legend issues
        pass


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


def is_valid_url(url):
    """
    Check if a URL is valid using a regular expression.
    
    Parameters:
    ----------
    url (str): The URL to validate.
    
    Returns:
    -------
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
    ----------
    url (str): The URL of the PDF file.
    output_path (str): The path where the downloaded PDF will be saved.
    """
    wget.download(url, out=output_path)
    


def download_pdf_urllib(url, output_path):
    """
    Download a PDF file from the given URL using urllib.

    Parameters:
    ----------
    url (str): The URL of the PDF file.
    output_path (str): The path where the downloaded PDF will be saved.
    """
    urllib.request.urlretrieve(url, output_path)    


def extract_text_from_pdf(pdf_path):
    """
    Extract the text from the PDF document.

    Parameters
    ----------
    pdf_path : str
        The path to the PDF file from which text is to be extracted.

    Returns
    -------
    text : str
        The extracted text from the PDF document.
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
    Extract the preceding and following characters around the found text.

    Parameters
    ----------
    text : str
        The text in which to search for the specified text.
    search_text : str
        The text to search for within the extracted text.
    context_length : int, optional
        The number of characters to extract before and after the found text. 
        The default is 400.

    Returns
    -------
    int
        The starting index of the found text within the extracted text.
    str
        The context around the found text, including the preceding and following characters.
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
    url : str
        The URL of the PDF file to be downloaded.
    output_path : str
        The path where the downloaded PDF will be saved.

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
    ----------
    directory : str 
        The path to the directory containing the files to be renamed.
    """
    for filename in os.listdir(directory):
        decoded_filename = urllib.parse.unquote(filename)
        if filename != decoded_filename:
            original_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, decoded_filename)
            os.rename(original_file_path, new_file_path)
            print(f'Renamed: {filename} -> {decoded_filename}')


def check_string_in_list(row: pd.Series, true: str, choice: str) -> str:
    """
    Function to check if the string 'true' is in the list 'choice'.

    Parameters
    ----------
    row : pd.Series
        The row of a DataFrame being processed.
    true : str
        The column name in the DataFrame whose value is to be checked.
    choice : str
        The column name in the DataFrame containing the list to check against.

    Returns
    -------
    str
        The value from the 'true' column if it is found in the 'choice' column list, otherwise np.nan.
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
    Plots the Precision-Recall curves for a multiclass classification problem.
    This function generates a plot that includes:
    - Precision-Recall curves for each class.
    - An average Precision-Recall curve over all classes.
    - Reference iso-F1 score contours.
    Parameters:
    - y_score (array-like): Estimated probabilities or decision function.
    - y_true_untransformed (array-like): True labels before transformation.
    - class_list (list): List of class labels.
    - classifier_name (str): Name of the classifier for labeling the plot.
        
    References:
    ----------
    The function is based on examples from scikit-learn and OpenAI's cookbook.
    [1] https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    [2] https://github.com/openai/openai-cookbook/blob/main/examples/utils/embeddings_utils.py
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
