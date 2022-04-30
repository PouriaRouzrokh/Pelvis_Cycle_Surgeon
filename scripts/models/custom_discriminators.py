#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

# Standard built-in modules
from collections import OrderedDict
import os
import sys
import warnings

# Third-party modules
import timm
import torch

# Local modules
# Adding the root direcory of the project to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
sys.path.append(root_path)

#-------------------------------------------------------------------------------
# Configurations
#-------------------------------------------------------------------------------
warnings.filterwarnings("ignore") 

#-------------------------------------------------------------------------------
# Custom Discriminators
#-------------------------------------------------------------------------------

#---------------------------------------
#- EffnetDiscriminator

class EffNetDiscriminator(torch.nn.Module):
    """Discriminator model with the EfficinentNet B0 stem."""
    def __init__(self, cnn_branch_block: int = 1):
        """Class constructor.

        Args:
            cnn_branch_block (int): the number of convolutional blocks to 
                collect.
        """
        super().__init__()
        cnn_frame_model = timm.create_model(
            'efficientnet_b0', pretrained=False, 
            num_classes=1, in_chans=1,  
            act_layer=torch.nn.LeakyReLU,
            norm_layer=torch.nn.InstanceNorm2d)
        
        stem_cnn = torch.nn.Sequential(OrderedDict([
            ("conv_stem", cnn_frame_model.conv_stem),
            ("bn1", cnn_frame_model.bn1),
            ("act1",cnn_frame_model.act1),
        ]))
        
        assert len(cnn_frame_model.blocks)>0, \
            "Only block based models are currently supported."
        assert cnn_branch_block <= len(cnn_frame_model.blocks), \
            f"The model has only {len(cnn_frame_model.blocks)} blocks."
        
        for i in range(cnn_branch_block):
            stem_cnn.add_module(name=f"cnn_block_{i}", 
                                module=cnn_frame_model.blocks[i])

        del cnn_frame_model
        
        self.model = torch.nn.Sequential(OrderedDict([
            ('stem', stem_cnn,),
            ('hard_bn1', torch.nn.BatchNorm2d(16)),
            ('head_act1', torch.nn.LeakyReLU(0.2)),
            ('head_conv1', torch.nn.Conv2d(16, 16, kernel_size=4, stride=2, 
                                           padding=1, padding_mode='reflect')),
            ('hard_bn2', torch.nn.BatchNorm2d(16)),
            ('head_act2', torch.nn.LeakyReLU(0.2)),
            ('head_conv2', torch.nn.Conv2d(16, 1, kernel_size=4, stride=2, 
                                           padding=1, padding_mode='reflect')),
            ('head_act3', torch.nn.Sigmoid())])
        )
                    
    def __call__(self, x):
        """ Forward pass.
        
        Args:
         x (torch.Tensor): input tensor.
         
        Returns:
          output (torch.Tensor): output tensor.
        """
        return self.model(x)
    
#---------------------------------------
#- ResNetDiscriminator
    
class ResNetDiscriminator(torch.nn.Module):
    """Discriminator model with the  ResNet18 stem."""
    def __init__(self, cnn_branch_block: int = 1):
        """Class constructor.

        Args:
            cnn_branch_block (int): the number of convolutional blocks to 
                collect.
        """
        super().__init__()
        cnn_frame_model = timm.create_model(
            'resnet18', pretrained=False, 
            num_classes=1, in_chans=1,  
            act_layer=torch.nn.LeakyReLU)
        
        stem_cnn = torch.nn.Sequential(OrderedDict([
            ("conv1", cnn_frame_model.conv1),
            ("bn1", cnn_frame_model.bn1),
            ("act1",cnn_frame_model.act1),
            ("maxpool1",cnn_frame_model.maxpool),
        ]))
        
        assert 4 >= cnn_branch_block >= 1, "cnn_branch_block must be in [1, 4]"
        
        for i in range(1, cnn_branch_block+1):
            stem_cnn.add_module(name=f"layer{i}", 
                                module=eval(f'cnn_frame_model.layer{i}'))

        stem_channels = self.figure_number_of_channels(stem_cnn)
        del cnn_frame_model
                
        self.model = torch.nn.Sequential(OrderedDict([
            ('stem', stem_cnn,),
            ('hard_bn1', torch.nn.BatchNorm2d(stem_channels)),
            ('head_act1', torch.nn.LeakyReLU(0.2)),
            ('head_conv1', torch.nn.Conv2d(stem_channels, 128, kernel_size=4, 
                                           stride=2, padding=1, 
                                           padding_mode='reflect')),
            ('hard_bn2', torch.nn.BatchNorm2d(128)),
            ('head_act2', torch.nn.LeakyReLU(0.2)),
            ('head_conv2', torch.nn.Conv2d(128, 256, kernel_size=4, 
                                stride=2, padding=1, 
                                padding_mode='reflect')),
            ('hard_bn3', torch.nn.BatchNorm2d(256)),
            ('head_act3', torch.nn.LeakyReLU(0.2)),
            ('head_conv3', torch.nn.Conv2d(256, 1, kernel_size=1, 
                    stride=1, padding=0, padding_mode='reflect')),
            ('head_act4', torch.nn.Sigmoid())])
        )
                    
    def figure_number_of_channels(self, stem_cnn: torch.nn.Sequential):
        x = torch.zeros(1, 1, 256, 256)
        y = stem_cnn(x)
        return y.shape[1]
    
    def __call__(self, x):
        """ Forward pass.
        
        Args:
         x (torch.Tensor): input tensor.
         
        Returns:
          output (torch.Tensor): output tensor.
        """
        return self.model(x)