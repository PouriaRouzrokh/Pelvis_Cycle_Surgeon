#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

# Standard built-in modules
from typing import Callable
import warnings

# Third-party modules
import torch
import torchvision

# Local modules
from utils.pytorch_utils import make_determinate

#-------------------------------------------------------------------------------
# Configurations
#-------------------------------------------------------------------------------

warnings.filterwarnings("ignore") 

#-------------------------------------------------------------------------------
# Generating the model
#-------------------------------------------------------------------------------

#---------------------------------------
#- F: build_model

def build_model(arch: str = 'vgg16', 
                     pretrained: bool = False, 
                     *args, **kwargs) -> Callable:
    """Build a resnet or vgg model using PyTorch.
    
    Args:
        arch (str): baseline architecture of the model that could be called 
            using torchvision.models.arch command. Defaults to vgg16.
        pretrained (bool): whether or not to use pretrained weights. Defaults to 
            True.
    
    Raises:
        ValueError: raises an error if the user asks for an unsupported arch.
    
    Returns:
        model (nn.Module): built pytorch module.
    """  
    if 'random_seed' in kwargs:
        if kwargs['random_seed'] is not None:
            make_determinate(kwargs['random_seed'])
    
    # Load a model with the user-specified architecture from torch
    if 'vgg' not in arch and 'alexnet' not in arch and 'resnet' not in arch:
        raise ValueError ('Only resnet, vgg or alexnet models can be loaded!')
    else:
        try:
            model = eval(f'torchvision.models.{arch}(pretrained={pretrained})')
        except:
            raise ValueError ('The name of the architecture is not valid!')

    # Replace the final fully conntected layer of the model.
    # The VGG network has no FC layer, so we directly change its final layer.
    if 'vgg' in arch or 'alexnet' in arch:
        model.classifier._modules['6'] = torch.nn.Linear(4096, 2)
    else: 
        num_in_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_in_features, 2)
    return model