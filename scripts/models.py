#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

# Standard built-in modules
from typing import Callable
import warnings

# Third-party modules
import torch
import torch.nn as nn
import torchvision

# Local modules
from utils.pytorch_utils import make_determinate

#-------------------------------------------------------------------------------
# Configurations
#-------------------------------------------------------------------------------

warnings.filterwarnings("ignore") 

#-------------------------------------------------------------------------------
# Generating the Baseline CycleGAN Model.
# See: https://www.youtube.com/watch?v=4LktBHGCNfw
#-------------------------------------------------------------------------------

#---------------------------------------
#- Discriminator 

#---------------------------------------
#-- C: CGDiscConvBLock 

class CGDiscConvBLock(nn.Module):
    """Discriminator block for BaseCGDiscriminator."""
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """ Class constructor.
        
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            stride (int): stride of the convolution.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, 
                      stride=stride, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        """ Forward pass.
        
        Args:
         x (torch.Tensor): input tensor.
         
        Returns:
          output (torch.Tensor): output tensor.
        """
        output = self.conv(x)
        return output

#---------------------------------------
#-- C: CGDiscriminator 

class CGDiscriminator(nn.Module):
    """Discriminator model for the original Cycle GAN."""
    def __init__(self, in_channels: int = 3, 
                 features: list = [64, 128, 256, 512]):
        """Class constructor.

        Args:
            in_channels (int, optional): number of input channels. 
                Defaults to 3.
            features (list, optional): number of kernels in each layer. 
                Defaults to [64, 128, 256, 512].
        """
        super().__init__()
        
        # The initial layer which does not have instance norm.
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, 
                      padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        
        # All other layers.
        layers = list()
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(CGDiscConvBLock(in_channels, 
                                          feature, 
                                          stride = 1 if feature==features[-1] \
                                              else 2))
            in_channels = feature
        
        # Adding a final layer and stacking all layers as the final model.
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1,
                                padding=1, padding_mode='reflect'))
        self.model= nn.Sequential(*layers)
        
    def forward(self, x):
        """ Forward pass.
        
        Args:
         x (torch.Tensor): input tensor.
         
        Returns:
          output (torch.Tensor): output tensor.
        """
        x = self.initial_conv(x)
        output = torch.sigmoid(self.model(x))
        return output
    
#---------------------------------------
#- Generator  

#---------------------------------------
#-- C: CGGenConvBLock 

class CGGenConvBLock(nn.Module):
    """Generator block for the BaseCGGenerator."""
    def __init__(self, in_channels: int, out_channels: int, down: bool = True,
                 use_act: bool = True, **kwargs):
        """Class constructor.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            down (bool, optional): whether to down sample. Defaults to True.
            use_act (bool, optional): whether to use the activation function. 
                Defaults to True.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      padding_mode='reflect', **kwargs) if down else \
                        nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )
        
    def forward(self, x):
        """ Forward pass.
        
        Args:
         x (torch.Tensor): input tensor.
         
        Returns:
          output (torch.Tensor): output tensor.
        """
        output = self.conv(x)
        return output
    
#---------------------------------------
#-- C: CGGenResBLock 

class CGGenResBLock(nn.Module):
    """Residual block for the BaseCGGenerator."""
    def __init__(self, channels: int):
        """Class constructor.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            down (bool, optional): whether to down sample. Defaults to True.
            use_act (bool, optional): whether to use the activation function. 
                Defaults to True.
        """
        super().__init__()
        self.block = nn.Sequential(
            CGGenConvBLock(channels, channels, kernel_size=3, padding=1),
            CGGenConvBLock(channels, channels, kernel_size=3, padding=1,
                            use_act=False)
        )
        
    def forward(self, x):
        """ Forward pass.
        
        Args:
         x (torch.Tensor): input tensor.
         
        Returns:
          output (torch.Tensor): output tensor.
        """
        return x + self.block(x)
    
#---------------------------------------
#-- C: CGGenerator

class CGGenerator(nn.Module):
    """Generator model for the original Cycle GAN."""
    def __init__(self, img_channels: int = 3, num_features: int = 64,
                 num_residuals: int = 9):
        """Class constructor.

        Args:
            img_channels (int, optional): _description_. Defaults to 3.
            num_residuals (int, optional): _description_. Defaults to 9.
        """
        super().__init__()
        
        # The initial layer which does not have instance norm.
        self.initial_conv = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, 
                      stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )
        
        # Down sampling layers.
        self.down_blocks = nn.ModuleList(
            [
                CGGenConvBLock(num_features, num_features*2, down=True,
                               kernel_size=3, stride=2, padding=1),
                CGGenConvBLock(num_features*2, num_features*4, down=True,
                               kernel_size=3, stride=2, padding=1),
            ]
        )
        
        # Residual blocks.
        self.residual_blocks = nn.Sequential(
            *[CGGenResBLock(num_features*4) for _ in range(num_residuals)]
        )
        
        # Up sampling layers.
        self.up_blocks = nn.ModuleList(
            [
                CGGenConvBLock(num_features*4, num_features*2, down=False,
                               kernel_size=3, stride=2, padding=1, 
                               output_padding=1),
                CGGenConvBLock(num_features*2, num_features, down=False,
                               kernel_size=3, stride=2, padding=1,  
                               output_padding=1),  
            ]
        )
        
        # The last layer to generate an image.
        self.last_conv = nn.Conv2d(num_features, img_channels, kernel_size=7,
                                   stride = 1, padding=3, 
                                   padding_mode='reflect')
        
    def forward(self, x):
        """ Forward pass.
        
        Args:
         x (torch.Tensor): input tensor.
         
        Returns:
          output (torch.Tensor): output tensor.
        """
        x = self.initial_conv(x)
        for down_block in self.down_blocks:
            x = down_block(x)
        x = self.residual_blocks(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        output = torch.tanh(self.last_conv(x))
        return output      