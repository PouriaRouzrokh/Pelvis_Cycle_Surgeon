#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

# Standard built-in modules
from typing import Union
import warnings

# Third-party modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
import torch

#-------------------------------------------------------------------------------
# Configurations
#-------------------------------------------------------------------------------

warnings.filterwarnings("ignore") 

#-------------------------------------------------------------------------------
# Custom metric functions
#-------------------------------------------------------------------------------

#-----------------------------------
# - F: plot_confusion_matrix

def plot_confusion_matrix(preds: Union[torch.Tensor, list[int], np.ndarray], 
                          labels: Union[torch.Tensor, list[int], np.ndarray], 
                          classes: list[str]) -> matplotlib.figure.Figure:
    """Plot a confusion matrix for a classifier.

    Args:
        preds (Union[torch.Tensor, list[int], np.ndarray]): 
            a tensor received preds with size: (N, C).
        labels (Union[torch.Tensor, list[int], np.ndarray]): 
            a tensor of recived labels with size: (N,).
        classes (List[str]): a list of class names.
    
    Raises:
        TypeError: if preds, labels or classes are not of the correct type.

    Returns:
        fig (matplotlib.figure.Figure): a matplotlib figure of the confusion
            matrix.
    """
    
    # Checking the typing of 'labels'
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)
    elif isinstance(labels, np.ndarray):
        pass
    else:
        raise TypeError("'labels' must be a torch.Tensor, a ndarray or a list.")
    
    # Checking the typing of 'preds'
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    elif isinstance(preds, list):
        preds = np.array(preds)
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError("'preds' must be a torch.Tensor, a ndarray or a list.")
    
    # Checking the typing of 'classes'
    if not isinstance(classes, list):
        raise TypeError("'classes' must be a list of strings.")
    
    # Plotting the confusion matrix and returning the figure
    cm = sklearn.metrics.confusion_matrix(labels, preds)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)
    return fig