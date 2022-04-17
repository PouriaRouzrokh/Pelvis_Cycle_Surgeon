#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

# Standard built-in modules
import os
import random
import shutil
import warnings

# Third-party modules
import monai as mn
import numpy as np
import torch
from tqdm import tqdm

#-------------------------------------------------------------------------------
# Configurations
#-------------------------------------------------------------------------------

warnings.filterwarnings("ignore") 

#-------------------------------------------------------------------------------
# General functions
#-------------------------------------------------------------------------------

#-----------------------------------
# - F: make_determinate

def make_determinate(random_seed):
    """Use a random_seed to enable deterministic programming for Pytorch, Moani, 
      and Numpy and.
    
    Args:
        random_seed (int): a random_seed to be used by different libraries.
    """
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    mn.utils.misc.set_determinism(seed=random_seed)

#-------------------------------------------------------------------------------
# Data helper functions
#-------------------------------------------------------------------------------

#-----------------------------------
# - F: get_data_stats

def get_data_stats(dataset:torch.utils.data.Dataset, 
                   img_key:str, num_channels:int = 1)->None:
    """_summary_

    Args:
        dataset (torch.utils.data.Dataset): the dataset to be used.
        img_key (str): the image key in the MONAI Dataset.
        num_channels (int, optional): number of channels in input images. 
            Defaults to 1.
    """
    pixels_sum=torch.zeros(num_channels)
    pixels_count=torch.zeros(num_channels)
    sum_squared_err=torch.zeros(num_channels)
    
    # Measuring the mean.
    for i,b in enumerate(tqdm(dataset)):
        image = b[img_key]
        pixels_sum = pixels_sum+image.sum((1,2))
        pixels_count = pixels_count+torch.tensor(
            [image.shape[1]*image.shape[2]]*num_channels)
    mean = pixels_sum/pixels_count

    # Measuring the STD.
    for i,b in enumerate(tqdm(dataset)):
        image = b[img_key].reshape(-1,num_channels)
        sum_squared_err = sum_squared_err + ((image - mean).pow(2)).sum()
    std = torch.sqrt(sum_squared_err / pixels_count)

    print("Final Mean:",mean)
    print("Final Std:",std)

#-----------------------------------
# - F: one_hot_encode

def one_hot_encode(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """A function to one-hot encode the true labels, while smoothing them if 
    requested.

    Args:
        true_labels (torch.Tensor): the input true labels.
        classes (int): number of classes.
        smoothing (float, optional): the magnitutde of label smoothing. 
            Defaults to 0.0.

    Returns:
        true_dist: the encoded true labels.
    """
    assert 0 <= smoothing < 1, 'smoothing must be between 0 and 1!'
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    if label_shape == true_labels.size():
        with torch.no_grad():
            true_dist = torch.where(true_labels==1.0, confidence, smoothing)
    else:
        with torch.no_grad():
            true_dist = torch.empty(size=label_shape, device=true_labels.device)
            true_dist.fill_(smoothing / (classes - 1))
            true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist

#-------------------------------------------------------------------------------
# Model helper functions
#-------------------------------------------------------------------------------

#-----------------------------------
# - F: save_checkpoint

def save_checkpoint(state: dict,  
                    is_best: bool = True,
                    checkpoint_dir: str = 'weights',
                    add_text: str = ''):
    """Saves the model in master process and loads it everywhere else.
    
    Args:
        state (dict): the state to be saved. Includes the model's state_dict,
            the epoch, and the step
        is_best (bool): whether or not this checkpoint has achieved the best 
            performance so far.
        checkpoint_dir (str, optional): a directory to save the checkpoint. 
            Defaults to None.
        add_text (str, optional): additional text to be added to the 
            saved model's name.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)   
    epoch = state['epoch']
    step = state['step'] 
    checkpoint_path = os.path.join(checkpoint_dir, 
                        f"checkpoint_epoch{epoch}_step{step}_{add_text}.pt")
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(checkpoint_dir, 
                                                f"best_checkpoint.pt"))
        
#-----------------------------------
# - F: load_weights
    
def load_weights(model: torch.nn.Module, 
                 weight_path: str) -> torch.nn.Module:
    """Load the weights of a model. This function is used to load the weights
    even if the target model is not exactly same as the source model used for
    saving the weights. Only the weights with the same names are loaded.

    Args:
        model (torch.nn.Module): the source model.
        weight_path (str): the target weights. Defaults to None.

    Returns:
        torch.nn.Module: the model with loaded weights.
    """
    weights = torch.load(weight_path)
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    model_dict.update(weights) 
    model.load_state_dict(model_dict)
    return model
