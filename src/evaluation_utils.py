# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:38:16 2024

@author: becker_nic

Evaluation Utils
"""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import numpy as np
import nltk

from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, multilabel_confusion_matrix, classification_report, confusion_matrix, ConfusionMatrixDisplay
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_bleu(reference, hypothesis):
    """
    Function to compute BLEU score

    Parameters
    ----------
    reference : TYPE
        DESCRIPTION.
    hypothesis : TYPE
        DESCRIPTION.

    Returns
    -------
    score : TYPE
        DESCRIPTION.

    """
    reference = [reference.split()]  # Reference should be a list of lists
    hypothesis = hypothesis.split()
    smoothing_function = SmoothingFunction().method1
    score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)
    return score

def stratified_sample_df(df, col, n_samples, random_state=0):
    """
    Docstring generated with Code Copilot.
    Perform stratified sampling on a DataFrame, ensuring up to a specified number of samples from each group.

    Parameters:
    df (pandas.DataFrame): 
        The input DataFrame.
    col (str): 
        The column name to stratify by.
    n_samples (int): 
        The maximum number of samples to draw from each group.

    Returns:
    df_ (pandas.DataFrame): 
        A DataFrame containing the stratified samples.
    
    The function works as follows:
    1. Group the DataFrame by the specified column.
    2. For each group, draw up to `n_samples` samples. If a group has fewer rows than `n_samples`, all rows are sampled.
    3. Use `random_state=0` to ensure reproducibility of the sampling process.
    4. Combine the samples from each group into a single DataFrame.

    """
    df_ = df.groupby(col, group_keys=False).apply(lambda x: x.sample(min(len(x), n_samples), random_state=random_state))
    return df_
    
def balanced_stratified_sample_df(df, col, n_samples, random_state=0):
    """
    From: https://stackoverflow.com/a/53615691
    Docstring generated with Code Copilot.
    
    Perform stratified sampling on a DataFrame, ensuring an equal number of samples from each group.

    Parameters:
    df (pandas.DataFrame): 
        The input DataFrame.
    col (str): 
        The column name to stratify by.
    n_samples (int): 
        The number of samples to draw from each group.

    Returns:
    df_ (pandas.DataFrame): 
        A DataFrame containing the stratified samples.
    
    The function works as follows:
    1. Determine the number of samples to draw from each group. 
       This is the minimum of `n_samples` and the size of the smallest group.
    2. Group the DataFrame by the specified column and apply a sampling function 
       to each group, ensuring `n` samples are drawn from each group.
    3. Reset the index to ensure the sampled DataFrame has a flat index.
    """
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n, random_state=random_state))
    df_.index = df_.index.droplevel(0)
    
    return df_ 


def create_pie_chart(ax, metric_value, metric_name):
    '''
    Function to create a pie chart for a given metric

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    metric_value : TYPE
        DESCRIPTION.
    metric_name : TYPE
        DESCRIPTION.
    colors : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    sizes = [metric_value, 1 - metric_value]
    labels = [metric_name, '']
    colors = [sns.color_palette('dark')[2] ,sns.color_palette('pastel')[7]]
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=0)
    ax.set_title(f'{metric_name} - Weighted Average')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    

def plot_evaluation(data: pd.DataFrame(), target_column: str, predicted_column: str, labels: list, title="Confusion Matrix" ):
    """
    Evaluate classification model performance by calculating accuracy, precision, and recall, and visualize results with pie charts and a confusion matrix.
    Docstring generated with Code Copilot
    
    Parameters:
    data (pd.DataFrame): 
        DataFrame containing target and predicted columns.
    target_column (str): 
        Name of the target column in the DataFrame.
    predicted_column (str): 
        Name of the predicted column in the DataFrame.
    title (str): 
        Title for the confusion matrix plot.
    labels (list): 
        List of class labels for evaluation.

    Returns:
    report_df (pd.DataFrame): 
        Classification report as a DataFrame.
    """
    data[target_column] = data[target_column].fillna('-').astype(str)
    data[predicted_column] = data[predicted_column].fillna('-').astype(str)
    
    # accuracy
    accuracy_norm = accuracy_score(data[target_column], data[predicted_column])
    accuracy = accuracy_score(data[target_column], data[predicted_column], normalize=False)
    
    print('accuracy_norm: ', accuracy_norm)
    print('accuracy: ', accuracy)
    
    # precision and recall scores
    precision = precision_score(data[target_column], data[predicted_column], labels=labels, average='weighted')
    recall = recall_score(data[target_column], data[predicted_column], labels=labels, average='weighted')
    
    print('precision: ', precision)
    print('recall: ', recall)
    
    ## Visualization
    
    # Create separate pie charts for precision and recall
    fig, axes = plt.subplots(1, 3, figsize=(12, 5)) 
    
    # Accuracy pie chart
    labels_accuracy = ['Accuracy Score', '']
    scores_accuracy = [accuracy_norm, 1 - accuracy_norm if accuracy_norm is not None else None]
    colors_accuracy = ['green', 'red']
    
    axes[0].pie(scores_accuracy, labels=labels_accuracy, autopct='%1.2f%%', startangle=90, colors=colors_accuracy)
    #axes[0].set_title('Accuracy Score')
    
    # Precision pie chart
    labels_precision = ['Precision Score', '']
    scores_precision = [precision, 1 - precision if precision is not None else None]
    colors_precision = ['green', 'red']
    
    axes[1].pie(scores_precision, labels=labels_precision, autopct='%1.2f%%', startangle=90, colors=colors_precision)
    axes[1].set_title("Performance "+predicted_column)
    
    # Recall pie chart
    labels_recall = ['Recall Score', ' ']
    scores_recall = [recall, 1 - recall if recall is not None else None]
    colors_recall = ['green', 'red']
    
    axes[2].pie(scores_recall, labels=labels_recall, autopct='%1.2f%%', startangle=90, colors=colors_recall)
    #axes[2].set_title('Recall Score')
    
    plt.show()
    
    # Attributes for Confusion Matrix
    y_true = data[target_column]
    
    # wrong indicator compositions are converted to '-'
    y_pred = data[predicted_column]#.copy().apply(lambda x: '-' if x not in labels else x)
    
    ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, cmap="YlGn")

    plt.title(title)
    plt.xticks(rotation=45)
    
    plt.show()
    
    
    # Classification Report
    report = classification_report(data[target_column], data[predicted_column], labels=labels, target_names=labels)
    print(report)
    
    report_dict = classification_report(data[target_column], data[predicted_column], labels=labels, target_names=labels, output_dict=True)
    
    # Convert the dictionary to a DataFrame
    report_df = pd.DataFrame(report_dict)
    
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    # extract avg weighted values from classification report
    precision = report_df.loc['precision','weighted avg']
    recall = report_df.loc['recall', 'weighted avg']
    f1_score = report_df.loc['f1-score', 'weighted avg']
    
    # Create separate pie charts in the same figure
    create_pie_chart(axes[0], precision, 'Precision')
    create_pie_chart(axes[1], recall, 'Recall')
    create_pie_chart(axes[2], f1_score, 'F1 Score')

    plt.tight_layout()
    plt.show()
    
    return report_df


def plot_classification_report(data: pd.DataFrame(), target_column: str, predicted_column: str, target_names: list, title='Classification Report', figsize=(8, 6), dpi=70, save_fig_path=None, **kwargs):
    """
    Plot the classification report of sklearn
    From: https://stackoverflow.com/a/74080540
    
    Parameters
    ----------
    data : pandas DataFrame
        Dataframe containing data to be evaluated.
    target_column : str
        Name of Target Column.
    predicted_column : str
        Name of column containing predictions.
    target_names : list
        List containing possible the target labels.
    title : str, default = 'Classification Report'
        Plot title.
    fig_size : tuple, default = (8, 6)
        Size (inches) of the plot.
    dpi : int, default = 70
        Image DPI.
    save_fig_path : str, defaut=None
        Full path where to save the plot. Will generate the folders if they don't exist already.
    **kwargs : attributes of classification_report class of sklearn
    
    Returns
    -------
        fig : Matplotlib.pyplot.Figure
            Figure from matplotlib
        ax : Matplotlib.pyplot.Axe
            Axe object from matplotlib
    """    
    
    y_test = data[target_column]
    y_pred = data[predicted_column].astype(str)
    
    ### original ###
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
    clf_report = classification_report(y_test, y_pred, output_dict=True, **kwargs)
    keys_to_plot = [key for key in clf_report.keys() if key not in ('accuracy', 'macro avg', 'weighted avg')]
    df = pd.DataFrame(clf_report, columns=keys_to_plot).T
    #the following line ensures that dataframe are sorted from the majority classes to the minority classes
    df.sort_values(by=['support'], inplace=True) 
    
    #first, let's plot the heatmap by masking the 'support' column
    rows, cols = df.shape
    mask = np.zeros(df.shape)
    mask[:,cols-1] = True
 
    ax = sns.heatmap(df, mask=mask, annot=True, cmap="YlGn", fmt='.3g',
            vmin=0.0,
            vmax=1.0,
            linewidths=2, linecolor='white'
                    )
    
    #then, let's add the support column by normalizing the colors in this column
    mask = np.zeros(df.shape)
    mask[:,:cols-1] = True    
    
    ax = sns.heatmap(df, mask=mask, annot=True, cmap="YlGn", cbar=False,
            linewidths=2, linecolor='white', fmt='.0f',
            vmin=df['support'].min(),
            vmax=df['support'].sum(),         
            norm=mpl.colors.Normalize(vmin=df['support'].min(),
                                      vmax=df['support'].sum())
                    ) 
            
    plt.title(title)
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 360)
         
    if (save_fig_path != None):
        path = pathlib.Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path)
    
    return fig, ax