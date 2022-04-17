#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

# Standard built-in modules
import os
import re
from typing import Union
import warnings

# Third-party modules
import monai as mn
import numpy as np
from tqdm import tqdm
import torch

# Local modules
import models
from utils import monai_utils

#-------------------------------------------------------------------------------
# Configurations
#-------------------------------------------------------------------------------

warnings.filterwarnings("ignore") 

#-------------------------------------------------------------------------------
# Inference tools
#-------------------------------------------------------------------------------

#---------------------------------------
# - C: InferenceModel

class InferenceModel():
    """A class to perform inference on target data using a trained model. 
    Objects of this class are initialized with a model, or a valid architecture 
    and a path to the trained weights for that architecture. 
    The model is then loaded with the weights and can be further applied to 
    objects of interest for inference, given path to a single object, a list 
    of paths to multiple objects, or a path to a directory containing objects.
    """
    
    #---------------------------------------
    #-- M: __init__
    
    def __init__(
        self,
        model: torch.nn.Module = None,
        model_arch: str = None,
        weights_path: str = f'..{os.path.sep}weights',
        best_weight_criterion: str = 'val-loss',
        best_weight_mode: str = 'min',
        use_gpu: bool = True,
        gpu_id: int = 0,
        use_dp: bool = True,
        verbose: bool = True,
    ):
        """Initialize the model for inference.

        Args:
            model (torch.nn.Module, optional): an already loaded model. 
                Defaults to None.
            model_arch (str, optional): the architecture to use for building a 
                model. Defaults to None.
            weights_path (str, optional): path to a trained weights for the
                model or folder containing multiple weights. In the latter case,
                the class will look for the best model based on two other 
                arguments: "best_weight_criterion" and "best_weight_mode". 
                Defaults to 'weights'.
            best_weight_criterion (str, optional): the criterion to use for 
                finding the best weights if "weights_path" is a directory. 
                Defaults to 'val-loss'.
            best_weight_mode (str, optional): the mode to use for finding the
                the best weights if "weights_path" is a directory. 
                Defaults to 'min'.
            use_gpu (bool, optional): whether to use GPU for training. 
                Defaults to True.
            gpu_id (int, optional): ID of the single GPU to use for training. 
                Defaults to 0.
            use_dp (bool, optional): whether to use DataParallel for GPU 
                training. Defaults to True.
            verbose (bool, optional): whether to print log messages during the
                inference. Defaults to True.
        """
        self.verbose = verbose
        self.model_arch = model_arch
        self.weights_path = weights_path
        self.use_gpu = use_gpu
        self.use_dp = use_dp
        self.gpu_id = gpu_id
        self.best_weight_criterion = best_weight_criterion
        self.best_weight_mode = best_weight_mode
        if model == None: 
            self.model = self.load_model()
        else:
            model.eval()
            self.model = model
        if use_gpu:
            self.model = self.setup_gpu(self.model)
        else:
            self.model.cpu()
            
    #---------------------------------------
    #-- M: load_model
    
    def load_model(self) -> torch.nn.Module:
        """Load a model's architecture and weights to be used for inference.

        Raises:
        ValueError: if no model or architecture is specified.
            
        Returns:
            model (torch.nn.Module): Built model architecture with 
                loaded weights.
        """
        # Build the architecture of the model.
        if self.model_arch == None:
            raise ValueError('You must specify a model or an architecture!')
        model = models.build_model(self.model_arch)
        self.log(f'A model was built with {self.model_arch} architecture.')
        
        # Load the weights of the model.'
        if os.path.isfile(self.weights_path):
            assert self.weights_path.endswith('.pt'), \
                'Weights must be a .pt file!'
        elif os.path.isdir(self.weights_path):
            self.log(['Searching for the best weight path from the weights',
                      f'directory based on the "{self.best_weight_criterion}"',
                      'criterion.'])
            weights_dict = dict()
            for root, _, files in os.walk(self.weights_path):
                for file in files:
                    if self.best_weight_criterion in file:
                        weights_dict[os.path.join(root, file)] = \
                            float(re.findall(r"{}=(\d*\.\d+|\d+)".\
                                format(self.best_weight_criterion), 
                                       r'{}'.format(file))[0])
            if self.best_weight_mode == 'min':
                self.weights_path = min(weights_dict, 
                                       key=lambda x: weights_dict[x])
            elif self.best_weight_mode == 'max':
                self.weights_path = max(weights_dict, 
                                       key=lambda x: weights_dict[x])
        model.load_state_dict(torch.load(self.weights_path)['state_dict'])
        
        # Log and return the model.
        self.log(f'Weights for the model were loaded from: {self.weights_path}')
        model.eval()
        return model
    
    #---------------------------------------
    #-- M: setup_gpu
    
    def setup_gpu(self, model: torch.nn.Module) -> torch.nn.Module:
        """Set-up the GPU for inference.

        Args:
            model (torch.nn.Module): the loaded model to be used for inference.

        Returns:
            model (torch.nn.Module): a set-up model on GPU ready for inference.
        """
        assert torch.cuda.is_available(), \
            'You must have a GPU to use the GPU for inference!'
        if self.use_dp:
            self.log(['Using Data Parallel with all available GPUs.', 
                     '"gpu_id" will be ignored.'])
            model.cuda()
            model = torch.nn.DataParallel(model)
        else:
            model.cuda(self.gpu_id)
            torch.cuda.set_device(self.gpu_id)
        return model
    
    #---------------------------------------
    #-- M: get_transforms
    
    def get_transforms(self) -> mn.transforms.Compose:
        """Build the required transforms for inference. Using MONAI by default.

        Returns:
            transforms (mn.transforms.Compos): the pipe-lined transforms.
        """
        transforms = mn.transforms.Compose([
          mn.transforms.LoadImageD(keys="image"),
          mn.transforms.EnsureChannelFirstD(keys="image"),
          monai_utils.EnsureGrayscaleD(keys="image"),
          mn.transforms.ResizeD(keys="image", spatial_size=(224, 224)),
          mn.transforms.NormalizeIntensityD(keys="image"),
          monai_utils.TransposeD(keys="image", indices=[0, 2, 1]),
          mn.transforms.ToTensorD(keys=["image",]),
          mn.transforms.RepeatChannelD(keys="image", repeats=3),
          mn.transforms.SelectItemsd(keys=["image",])
          ])
        
        return transforms
    
    #---------------------------------------
    #-- M: build_dataset   
    
    def build_dataset(self, objects: Union[list, str],
                      transforms: mn.transforms.Compose
                      ) -> torch.utils.data.Dataset:
        """Build the required dataset for inference. Using MONAI by default.

        Args:
            objects (Union[list, str]): A signle string path or a list of 
                string paths, pointing to files or directories containing
                files to be used for inference.
                All objects should have one of the following extensions:
                ['nii', 'dcm', 'jpg', 'jpeg', 'png', 'npy']
            transforms (mn.transforms.Compose): the pipe-lined transforms.
            
        Raises:
            ValueError: if the input file does not have the expected extension.
            ValueError: if the 'objects' is not a list or a string.

        Returns:
            dataset (torch.utils.data.DataSet): the built dataset.
        """
        
        # Take care of situations where objects is a single string paths. 
        if isinstance(objects, str):
            
            # Path is pointing to a directory of objects.
            if os.path.isdir(objects):
                target_dir = objects
                object_list = list()
                for root, _, files in os.walk(target_dir):
                    for file in files:
                        if file.split('.')[-1] in \
                            ['nii', 'dcm', 'jpg', 'jpeg', 'png', 'npy']:
                            object_list.append({'image':
                                os.path.join(root, file)})
                self.log([f'{len(object_list)} objects were found',
                          f'from the source directory located at:', 
                          f'{target_dir}'])
            
            # Path is pointing to a single object.
            elif os.path.isfile(objects):
                if objects.split('.')[-1] in \
                        ['nii', 'dcm', 'jpg', 'jpeg', 'png', 'npy']:
                    object_list = [{'image': objects}]
                else:
                    raise ValueError('The input file must be a .nii, .dcm, '\
                                     '.jpg, .jpeg, .png or .npy file!')
        
        # Take care of situations where objects is a list of string paths. 
        elif isinstance(objects, list):
            object_list = list()
            for i, object in enumerate(objects):
                if not isinstance(object, str):
                    self.log([f'Found and ignored a non-string object in',
                              f'position {i} of the input list!'])
                    continue
                if object.split('.')[-1] not in \
                            ['nii', 'dcm', 'jpg', 'jpeg', 'png', 'npy']:
                    self.log([f'Found and ignored a non standard object path',
                              f' in position {i} of the input list!'])
                    continue
                object_list.append({'image': object})    
        else:
            raise ValueError('"objects" must be a string or a list!')
        
        # Build and return the dataset.
        dataset = mn.data.Dataset(object_list, transforms)
        self.log(f'The dataset was built with length: {len(dataset)}')
        return dataset
    
    #---------------------------------------
    #-- M: post_porocess  
    
    def post_porocess(self, logits: torch.Tensor) -> list[np.ndarray]:
        """Post-process the logits to get the predictions.
        
        Args:
            logits (torch.Tensor): the logits of the model.
            
        Returns:
            predictions (list): the list of post-processed predictions.
        """
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        preds = list(preds.cpu().numpy())
        return preds
    
    #---------------------------------------
    #-- M: predict
    
    def predict(
        self, 
        objects: Union[list, str],
        batch_size: int = 4,
        dl_num_workers: int = 4) -> tuple[list, list]:
        """The main method to apply the model for inference.

        Args:
            objects Union[list, str]:  A signle string path or a list of 
                string paths, pointing to files or directories containing
                files to be used for inference.
                All objects should have one of the following extensions:
                ['nii', 'dcm', 'jpg', 'jpeg', 'png', 'npy']
            batch_size (int): the batch size for inference, if "use_gpu" is
                set to True.
            dl_num_workers (int): the number of workers for data loading, if 
                "use_gpu" is set to True.
                
        Raises:
            ValueError: if the dataset is empty.

        Returns:
            processed_outputs, raw_outputs (tuple(list, list)): 
                a tuple of two lists, containing the processed and raw outputs
                of the model.
        """
        
        # Prepare the dataset, and check the device to run the model.
        transforms = self.get_transforms()
        dataset = self.build_dataset(objects, transforms)
        if len(dataset) == 0:
            raise ValueError('The dataset is empty!')
        elif len(dataset) < batch_size:
            self.log(['The batch size is larger than the dataset!', 
                      'Running inference on CPU...'])
            use_cpu_this_time = True
        else:
            use_cpu_this_time = False
        raw_outputs = list()
        
        # Making inference using CPU.
        if not self.use_gpu or use_cpu_this_time==True:
            self.log(f'Using CPU for inference!')
            with torch.no_grad():
                for data in tqdm(dataset):
                    image = data['image'].unsqueeze(0)
                    logit = self.model(image)
                    raw_outputs.append(logit)
            raw_outputs = torch.cat(raw_outputs)
            processed_outputs = self.post_porocess(raw_outputs)
            raw_outputs = raw_outputs.tolist()
        
        # Making inference using GPU.
        else:
            self.log([f'Using GPU for inference with DDP model = {self.use_dp}',
                      f', batch size: {batch_size}, and {dl_num_workers}',
                      'workers.'])
            test_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=dl_num_workers, pin_memory=True, drop_last=True)
            raw_outputs = list()
            processed_outputs = list()
            with torch.no_grad():
                for data in tqdm(test_loader):
                    images = data['image']
                    if self.use_dp:
                        images = images.cuda()
                    else:
                        images = images.cuda(self.gpu_id)
                    logits = self.model(images)
                    preds = self.post_porocess(logits)
                    for logit in logits:
                        raw_outputs.append(logit.cpu().tolist())
                    for pred in preds:
                        processed_outputs.append(pred)
        
        # Return the raw and processed outputs.
        return processed_outputs, raw_outputs
        
    #---------------------------------------
    #-- M: log
    
    def log(self, input_data: Union[list, str], joint_char: str = ' '):
        """Print a message in the console if self.verbose is set to True.
        Takes care of concatenating the input data into a single string, if a 
        list of string is given.

        Args:
            input_data (Union[list, str]): message or messages to be printed.
            joint_char (str, optional): what charactar to use for concatenating
                the strings in the input list if a list is given. 
                Defaults to ' '.
        """
        if self.verbose:
            if isinstance(input_data, list):
                input_data = f'{joint_char}'.join(input_data)
            print(f'---Inference logger: {input_data}')
                       
    