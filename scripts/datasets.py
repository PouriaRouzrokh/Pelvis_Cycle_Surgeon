#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

# Standard built-in modules
import warnings

# Third-party modules
import monai as mn
import numpy as np
import os
import pandas as pd
import torch

# Local modules
from utils import monai_utils

#-------------------------------------------------------------------------------
# Configurations
#-------------------------------------------------------------------------------

root_path = os.path.dirname(os.path.dirname(__file__))
warnings.filterwarnings("ignore")  

#-------------------------------------------------------------------------------
# Building datasets
#-------------------------------------------------------------------------------

#---------------------------------------
# - F: build_datasets

def build_datasets(
    data_index_path: str = \
        f'{root_path}{os.path.sep}data{os.path.sep}data_index.csv', 
    image_size: int = 224
    ) -> list[mn.data.Dataset] :
    """Use MONAI to build training, validation and test datasets. 
    
    Args:
        data_index_path (str, optional): the path to the data index csv file.
            Defaults to 'data/data_index.csv'.
        image_size (int, optional): the output image size for the dataloader, 
            which would be (image_size * image_size). Defaults to 224.

    Returns:
       train_dataset(mn.data.Dataset): training dataset.
       valid_dataset(mn.data.Dataset): validation dataset.
    """
    # Load the data index csv file.
    data_index = pd.read_csv(data_index_path)
   
    # Build the dataset_dict.
    data_dict = dict()
    label_dict = {'PNEUMONIA':1, 'NORMAL':0}
    for i, row in data_index.iterrows():
        set_list = data_dict.get(row['file_set'], list())
        set_list.append({'image': row['file_path'], 
                        'label': label_dict[row['file_label']]})
        data_dict[row['file_set']] = set_list

    # Build MONAI transforms.
    Aug_Ts = mn.transforms.Compose([
          mn.transforms.LoadImageD(keys="image"),
          mn.transforms.EnsureChannelFirstD(keys="image"),
          monai_utils.EnsureGrayscaleD(keys="image"),
          mn.transforms.ResizeD(keys="image", 
                                spatial_size=(image_size, image_size)),
          mn.transforms.NormalizeIntensityD(keys="image"),
          mn.transforms.RandRotateD(keys="image", mode="bilinear", 
                                    range_x=0.26, prob=0.5),
          mn.transforms.RandZoomD(keys="image", mode="bilinear"),
          monai_utils.TransposeD(keys="image", indices=[0, 2, 1]),
          mn.transforms.ToTensorD(keys=["image", "label"]),
          mn.transforms.RepeatChannelD(keys="image", repeats=3),
          mn.transforms.SelectItemsd(keys=["image", "label"])
          ])
    NoAug_Ts = mn.transforms.Compose([
          mn.transforms.LoadImageD(keys="image"),
          mn.transforms.EnsureChannelFirstD(keys="image"),
          monai_utils.EnsureGrayscaleD(keys="image"),
          mn.transforms.ResizeD(keys="image", 
                                spatial_size=(image_size, image_size)),
          mn.transforms.NormalizeIntensityD(keys="image"),
          monai_utils.TransposeD(keys="image", indices=[0, 2, 1]),
          mn.transforms.ToTensorD(keys=["image", "label"]),
          mn.transforms.RepeatChannelD(keys="image", repeats=3),
          mn.transforms.SelectItemsd(keys=["image", "label"])
          ])

    # Build MONAI datasets.
    train_dataset = mn.data.Dataset(data_dict['train'], transform=Aug_Ts)
    valid_dataset = mn.data.Dataset(data_dict['test'], transform=NoAug_Ts)

    return train_dataset, valid_dataset

#-------------------------------------------------------------------------------
# Oversampling tools
#-------------------------------------------------------------------------------

#---------------------------------------
# - F: build_datasets

def get_oversampling_sampler(dataset: mn.data.Dataset,
                            label_key: str = 'label',
                            )-> torch.utils.data.sampler.WeightedRandomSampler:
    """Create a sampler that oversamples the minority classes.

    Args:
        dataset (torch.utils.data.mn.data.Dataset): the input dataset (built as a 
            MONAI dictionary dataset.)
        label_key (str, optional): the key for data label in the MONAI 
            dictionary dataset. Defaults to 'label'.

    Returns:
        oversampling_sampler (torch.utils.data.sampler.WeightedRandomSampler): 
            the oversampling sampler that should be passed to DataLoader or the 
            DistributedProxySampler constructors.
    """
    # Load the 'class_weight_dict' or build one from the dataset.
    labels = np.array([data[label_key] for data in dataset.data])
    class_weight_dict = {}
    train_label_np = np.array(labels)
    class_labels, class_counts = np.unique(train_label_np, 
                                            return_counts=True)
    majority_class_count = max(class_counts)
    for i, class_label in enumerate(class_labels):
        class_weight_dict[class_label] = majority_class_count/class_counts[i]
    
    # Build the oversampling_sampler.
    sample_weights=[class_weight_dict[class_label] for class_label in labels]
    oversampling_sampler= torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(dataset), replacement=True)
    return oversampling_sampler

