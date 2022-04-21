#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

# Standard built-in modules
import os
import shutil
import warnings

# Third-party modules
import monai as mn
import numpy as np
import pandas as pd
import torch

# Local modules
from utils import monai_utils

#-------------------------------------------------------------------------------
# Configurations
#-------------------------------------------------------------------------------

root_path = os.path.dirname(os.path.dirname(__file__))
sep = os.path.sep
warnings.filterwarnings("ignore")  

#-------------------------------------------------------------------------------
# Building datasets
#-------------------------------------------------------------------------------

#---------------------------------------
# - C: PCSDataSet

class PCSDataSet(torch.utils.data.Dataset):
    """A class for loading a PyTorch dataset for training the CycleGAN
        model. This class will internally use MONAI PersistentDatasets.
        Inherits from torch.utils.data.Dataset.
    """
    def __init__(self, 
                 data_index_path: str = \
                     f'{root_path}{sep}data{sep}data_index.csv',
                 image_size: int = 224,
                 valid_fold: int = 0,
                 mode: str = 'train'):
        """Class constructor.

        Args:
            data_index_path (str, optional): path to the data_index.csv file. 
                Defaults to \f'{root_path}{sep}data{sep}data_index.csv'.
            image_size (int, optional): final size of the images. 
                Defaults to 224.
            valid_fold (int, optional): Number of the fold to be used for 
                validation. Fold -1 will always be the test set. All other folds
                build the training set together. Defaults to 0.
            mode (str, optional): Whether to build the training, validation, or
                test datasets. Defaults to 'train'.

        Raises:
            ValueError: If mode is not 'train', 'valid', or 'test'.
        """
        super().__init__()
        
        # Load the data index csv file and split it into pre- and post- 
        # dataframes.
        df = pd.read_csv(data_index_path)
        if mode == 'valid':
            df = df[df['Fold'] == valid_fold]        
        elif mode=='train': 
            df = df[~df["Fold"].isin([-1, valid_fold])]
        elif mode=='test':
            df = df[df['Fold'] == -1]
        else:
            raise ValueError('The "mode" should be "train", "valid" or "test"!')
        pre_df = df[df['STATE'] == 'PRE']
        post_df = df[df['STATE'] == 'POST']
        
        # Build MONAI transforms with augmentatiin.
        Aug_Ts = mn.transforms.Compose([
            monai_utils.LoadCropD(keys=["image", "crop_key"], dilation=50),
            monai_utils.PadtoSquareD(keys="image"),
            mn.transforms.ResizeD(keys="image", 
                                    spatial_size=(image_size, image_size)),
            mn.transforms.RandRotateD(keys="image", mode="bilinear", 
                                        range_x=0.26, prob=0.5),
            mn.transforms.RandZoomD(keys="image", mode="bilinear"),
            mn.transforms.ScaleIntensityD(keys="image"),
            monai_utils.TransposeD(keys="image", indices=[0, 2, 1]),
            mn.transforms.ToTensorD(keys=["image"]),
            mn.transforms.RepeatChannelD(keys="image", repeats=3),
            ])
        
        # Build MONAI transforms without augmentatiin.
        NoAug_Ts = mn.transforms.Compose([
            monai_utils.LoadCropD(keys=["image", "crop_key"], dilation=50),
            monai_utils.PadtoSquareD(keys="image"),
            mn.transforms.ResizeD(keys="image", 
                                    spatial_size=(image_size, image_size)),
            mn.transforms.ScaleIntensityD(keys="image"),
            monai_utils.TransposeD(keys="image", indices=[0, 2, 1]),
            mn.transforms.ToTensorD(keys=["image"]),
            mn.transforms.RepeatChannelD(keys="image", repeats=3),
            ])
        
        # Choose the appropriate transforms.
        if mode == 'train':
            Ts = Aug_Ts
        else:
            Ts = NoAug_Ts
        
        # Build data dictionaries to be fed into MONAI PersistentDatasets.
        pre_dict = [{'image': pre_df.iloc[i]['DICOM_Path'],
                     'crop_key': [pre_df.iloc[i]['X_min'],
                                  pre_df.iloc[i]['Y_min'],
                                  pre_df.iloc[i]['Width'],
                                  pre_df.iloc[i]['Height']]} \
            for i in range(len(pre_df))]
        post_dict = [{'image': post_df.iloc[i]['DICOM_Path'],
                      'crop_key': [post_df.iloc[i]['X_min'],
                                   post_df.iloc[i]['Y_min'],
                                   post_df.iloc[i]['Width'],
                                   post_df.iloc[i]['Height']]} \
            for i in range(len(post_df))]
        
        # Preparing the cache dirs.
        pre_cache_dir = f'{sep}scratch{sep}projects{sep}'\
                        f'm221279_Pouria{sep}PCS{sep}{mode}_pre_cache'
        post_cache_dir = f'{sep}scratch{sep}projects{sep}'\
                         f'm221279_Pouria{sep}PCS{sep}{mode}_post_cache'
        shutil.rmtree(pre_cache_dir, ignore_errors=True)
        shutil.rmtree(post_cache_dir, ignore_errors=True)
        os.makedirs(pre_cache_dir, exist_ok=True)
        os.makedirs(post_cache_dir, exist_ok=True)

        # Build MONAI PersistentDatasets.
        self.pre_dataset = mn.data.PersistentDataset(pre_dict, 
                                                     cache_dir=pre_cache_dir, 
                                                     transform=Ts)
        self.post_dataset = mn.data.PersistentDataset(post_dict, 
                                                     cache_dir=post_cache_dir, 
                                                     transform=Ts)
            
    def __len__(self):
        """Returns the length of the dataset."""
        return max(len(self.pre_dataset), len(self.post_dataset))
    
    def __getitem__(self, index: int) -> dict:
        """Returns the torch tensors for one pre-op Xray and one post-op Xray at 
        the given index.

        Args:
            index (int): The index to be used.

        Returns:
            {'pre': pre_img, 'post': post_img} (dict): A dict of torch tensors
                to be returned.
        """
        pre_index = index % len(self.pre_dataset)
        post_index = index % len(self.post_dataset)  
        pre_img = self.pre_dataset[pre_index]['image']
        post_img = self.post_dataset[post_index]['image']
        return {'pre': pre_img, 'post': post_img}

