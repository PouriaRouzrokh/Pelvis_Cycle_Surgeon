#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

# Standard built-in modules
import argparse
import datetime
import gc
import os
import random
import statistics
import shutil
import time
from typing import Callable, Iterable, Union
import warnings

# Third-party modules
import torch
import torchmetrics
from tqdm import tqdm
import wandb

# Local modules
import datasets
import models
import schedulers
from utils import ddp_utils
from utils import pytorch_utils

#-------------------------------------------------------------------------------
# Configurations
#-------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
root_path = os.path.dirname(os.path.dirname(__file__))
os.environ['WANDB_API_KEY'] = '...'
os.environ["WANDB_SILENT"] = "true"

#-------------------------------------------------------------------------------
# Parser
#-------------------------------------------------------------------------------

parser_arguments = {
    
    #---------------------------------------
    # - G: Naming conventions
    
    'project_name': {
        'default': 'PyTorch-DDP-Demo', 
        'type': str, 
        'help': 'Name of the project.',
        'action': 'store'
    },
    
    'exp_name': {
        'default': 'Exp-1', 
        'type': str, 
        'help': 'Name of the experiment.',
        'action': 'store'
    },
    
    'run_name': {
        'default': None, 
        'type': str, 
        'help': 'Name of the training run (default: the current datetime).',
        'action': 'store'
    },
    
    #---------------------------------------
    # - G: Model hyperparameters
    
    'pretrained': {
        'default':True, 
        'help': 'Use a pre-trained model.',
        'action': 'store_true'
    },
    
    'checkpoint_dir': {
        'default': None, 
        'type': str, 
        'help': 'Folder path for saving model checkpoints (default: none).',
        'action': 'store'
    },
    
    'resume': {
        'default': None, 
        'type': str, 
        'help': 'Folder path for loading model checkpoints (default: none).',
        'action': 'store'
    },
    
    #---------------------------------------
    # - G: Training hyperparameters

    'epochs': {
        'default': 2, 
        'type': int, 
        'help': 'Number of total epochs to run.',
        'action': 'store'
    },
    
    'start_epoch': {
        'default': 0, 
        'type': int, 
        'help': 'Manual epoch number (useful on restarts).',
        'action': 'store'
    },
    
    'batch_size': {
        'default': 64, 
        'type': int, 
        'help': 'Mini-batch size.',
        'action': 'store'
    },

    'lr': {
        'default': 0.01, 
        'type': float, 
        'help': 'Initial learning rate.',
        'action': 'store'
    },
    
    'momentum': {
        'default': 0.9, 
        'type': float, 
        'help': 'Momentum value.',
        'action': 'store'
    },
    
    'weight_decay': {
        'default': 1e-4, 
        'type': float, 
        'help': 'Weight decay value.',
        'action': 'store'
    },
    
    'early_stop_patience': {
        'default': 3, 
        'type': int, 
        'help': 'Number of epochs to wait before early stopping.',
        'action': 'store'
    },
    
    'log_freq': {
        'default': 2, 
        'type': int, 
        'help': 'Log frequency in steps. Pass -1 to log every epoch.',
        'action': 'store'
    },
    
    'oversample_train': {
        'default': True, 
        'help': 'Oversample the training set.',
        'action': 'store_true'
    },
    
    'oversample_valid': {
        'default': False, 
        'help': 'Oversample the validation set.',
        'action': 'store_true'
    },
    
    'seed': {
        'default': None, 
        'type': int, 
        'help': 'Random seed for initializing training.',
        'action': 'store'
    },
    
    'workers': {
        'default': 4, 
        'type': int, 
        'help': 'Number of data loading workers.',
        'action': 'store'
    },
    
    #---------------------------------------
    # - G: GPU configurations 

    'parallel_mode': {
        'default': 'ddp', 
        'type': str, 
        'help': 'Parallel model. values could be "ddp", "dp", or None.',
        'action': 'store'
    },

    'gpu': {
        'default': None, 
        'type': int, 
        'help': 'GPU id to use for training; e.g., 0.',
        'action': 'store'
    },
    
    'sync_batchnorm': {
        'default': True, 
        'help': 'Use torch.nn.SyncBatchNorm.convert_sync_batchnorm.',
        'action': 'store_true'
    },

    'n_nodes': {
        'default': 1, 
        'type': int, 
        'help': 'Number of nodes for distributed training.',
        'action': 'store'
    },

    'node_rank': {
        'default': 0, 
        'type': int, 
        'help': 'Node rank for distributed training.',
        'action': 'store'
    },

    'dist_url': {
        'default': "env://", 
        'type': str, 
        'help': 'The URL used to set up distributed training.',
        'action': 'store'
    },

    'dist_backend': {
        'default': "nccl", 
        'type': str, 
        'help': 'The distributed backend.',
        'action': 'store'
    },

    'master_addr': {
        'default': "127.0.0.1", 
        'type': str, 
        'help': 'The IP address of the host node.',
        'action': 'store'
    },

    'master_port': {
        'default': "29500", 
        'type': str, 
        'help': 'The port number for the host node.',
        'action': 'store'
    },

    'verbose_ddp': {
        'default': True, 
        'help': 'Print DDP status and logs.',
        'action': 'store_true'
    },
}

# build the parser.
parser = argparse.ArgumentParser()
for key, value in parser_arguments.items():
    parser.add_argument('--'+key, **value)

#-------------------------------------------------------------------------------
# Body of the script
#-------------------------------------------------------------------------------

#---------------------------------------
# - F: main

def main():
    """The main function to be executed. The ultimate goal of this function is 
    to check the status for parallel execution of training, and if needed spawn 
    the main_worker function to different GPUs by multiprocessing. it will also 
    do several initial sanity checks to start the training. 

    Raises:
        RuntimeError: raises an error if no GPU is detected.
    """
    args = parser.parse_args()
    if args.run_name is None:
        now = datetime.datetime.now()
        args.run_name = f'{args.exp_name}_{now.strftime("%Y%m%d_%H%M%S")}'
    
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU was detected!")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    if args.gpu is not None:
        args.parallel_mode = None
        warnings.warn('You have chosen a specific GPU. This will '
                      'completely disable data parallelism.')

    args.ngpus_per_node = torch.cuda.device_count()
    if args.parallel_mode == 'ddp':
        
        # Check if at least 2 GPUs are available.
        assert args.ngpus_per_node > 1, "You need at least 2 GPUs for DDP."
        assert torch.distributed.is_available(), \
            "torch.distribution is not available!"
        
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = args.ngpus_per_node * args.n_nodes
        
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        torch.multiprocessing.spawn(main_worker, 
                                    nprocs=args.ngpus_per_node, args=(args,))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, args)
    
#---------------------------------------
# - F: main_worker

def main_worker(gpu: int, args: argparse.Namespace):
    """The main worker function that is going to be spawned to each GPU.
    - Epoch loop will be defined in this function.
    - The model, loss, optimizer, and data loaders will be defined here.

    Args:
        gpu (int): the rank of the current gpu within the current node.
        args (argparse.Namespace): parsed arguments.

    Raises:
        ValueError: riases an error if parallel_model is not "dp" or "ddp".
    """
    
    #---------------------------------------
    #-- Global variables
    # Do not forget to remention variables like 'best_benchmark' as global 
    # variables inside 'train_one_epoch' and 'validate_one_epoch' functions. 
    # This is necessary as such variables are going to be re-assigned in those 
    # functions. This is not necessary to do for global bool variables that will 
    # not change in those functions or global list/dictionaries that will get 
    # new elements (but not be reassigned) in those functions.
    
    global is_base_rank 
    global is_ddp
    global train_step_logs
    global valid_step_logs
    global train_epoch_logs
    global valid_epoch_logs
    global checkpoint_dir
    global best_benchmark
    global early_stop_counter
    
    is_base_rank = bool()
    is_ddp = bool()
    train_step_logs = dict()
    valid_step_logs = dict()
    train_epoch_logs = dict()
    valid_epoch_logs = dict()
    best_benchmark = float()
    checkpoint_dir = str()
      
    # The best benchmark will keep the value of a benchmark to save the model
    # weights during the training. This will be defined in the validation step.
    best_benchmark = None
    early_stop_counter = 0
    
    #---------------------------------------
    #-- Model loading
    
    model = models.build_model('resnet18', pretrained=args.pretrained, 
                             random_seed=args.seed)
    if args.parallel_mode=='ddp' and args.sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    #---------------------------------------
    #-- Data parallel configurations
    
    # Distributed Data Parallel; we need to calculate the batch size for each
    # GPU manually. 
    if args.parallel_mode=='ddp':
        ddp_utils.setup_ddp(gpu, args)
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        args.workers = int((args.workers + 
                            args.ngpus_per_node-1) / args.ngpus_per_node)
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        is_base_rank = args.gpu == 0
        is_ddp = True
        store = torch.distributed.TCPStore(host_name = args.master_addr, 
                                           port = 1234,
                                           world_size = -1, 
                                           is_master = is_base_rank)
        # Set the base stores.
        if is_base_rank:
            store.set('early_stop', 'disabled')
    
    # Data Parallel; PyTorch will automatically divide and allocate batch_size 
    # to all available GPUs.
    elif args.parallel_mode=='dp':  
        model.cuda()
        model = torch.nn.DataParallel(model)
        is_base_rank = True
        is_ddp = False
        
    # Single GPU Training
    elif args.parallel_mode==None:
        torch.cuda.set_device(args.gpu)
        args.parallel_mode= None
        model = model.cuda(args.gpu)
        is_base_rank = True
        is_ddp = False
    
    # Unknown parallel mode
    else:
        raise ValueError('parallel_mode should be "dp" or "ddp".')
    
    # Reporting
    if is_base_rank:
        print('-'*80)
        print('Starting the training with: ')
        if args.parallel_mode in ['dp', 'ddp']:
            print(f'Number of nodes: {args.n_nodes}).')
            print(f'Number of GPUs per node: {args.ngpus_per_node}')
        else:
            print(f'GPU: {args.gpu}')
        print('-'*80)
    
    #---------------------------------------
    #-- Checkpoint directory
    
    if is_base_rank:
        if args.checkpoint_dir is None:
            checkpoint_dir = os.path.join(f'{root_path}{os.path.sep}weights', 
                                        args.exp_name, args.run_name) 
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir)
    
    #---------------------------------------
    #-- Resuming from a checkpoint
    
    if args.resume:
        if os.path.isfile(args.resume):
            if is_base_rank:
                print(f"=> loading checkpoint '{args.resume}'")
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f'cuda:{args.gpu}'
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            if is_base_rank:
                print(f"=> loaded checkpoint '{args.resume}' "
                    f"(epoch {checkpoint['epoch']})")
        else:
            if is_base_rank:
                print("=> no checkpoint found at '{}'".format(args.resume))
    
    #---------------------------------------
    #-- Datasets & data loaders
    # The minority classes in training and validation sets will be oversampled 
    # if specified in the arguments.
    # In dataloaders, shuffle should be set to False in case of DDP.
    
    train_dataset, valid_dataset = datasets.build_datasets()
    
    if args.oversample_train:
        train_sampler = datasets.get_oversampling_sampler(train_dataset)
    else:
        train_sampler = None
    if args.oversample_valid:
        valid_sampler = datasets.get_oversampling_sampler(valid_dataset)
    else:
        valid_sampler = None
    
    if is_ddp: 
        if args.oversample_train:
            train_sampler = ddp_utils.DistributedProxySampler(train_sampler)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
        if args.oversample_valid:
            valid_sampler = ddp_utils.DistributedProxySampler(valid_sampler)
        else:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), 
        num_workers=args.workers,
        pin_memory=True, sampler = train_sampler,
        drop_last=True
        )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=(valid_sampler is None), 
        num_workers=args.workers,
        pin_memory=True, sampler = valid_sampler,
        drop_last=True
        )
    if is_base_rank:
        print('The dataloaders are built.')
    
    #---------------------------------------
    #-- Loss
    
    criterion = torch.nn.CrossEntropyLoss()
    if args.gpu is None:
        criterion.cuda()
    else:
        criterion.cuda(args.gpu) 
    
    #---------------------------------------
    #-- Optimizer & scheduler
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # Define the learning rate scheduler using CosineAnnealingWarmupRestarts.
    # Alternatively, use torch.optim.lr_scheduler schedulers.
    # If no scheduler is needed, pass it as None.
    scheduler = None
    """
    scheduler = schedulers.CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=40,
        cycle_mult=1.0,
        max_lr=0.01,
        min_lr=0.001,
        warmup_steps=10,
        gamma=0.5
        )
    """
    
    #---------------------------------------
    #-- WandB initilization
    
    if is_base_rank:
        wandb.init(
        project = args.project_name, 
        group = args.exp_name, 
        name = args.run_name,
        notes = None,
        config = args,
        mode = 'online',
        save_code = True,
        )
        
        # Define which metrics should be plotted against "epochs" as the X axis.
        epoch_metrics = ['train_epoch_loss', 'valid_epoch_loss', 
                         'train_epoch_acc', 'valid_epoch_acc']
        wandb.define_metric("epochs")
        for metric in epoch_metrics:
            wandb.define_metric(metric, step_metric="epochs")
    
    #---------------------------------------
    #-- Epoch loop
    
    # Enable cudnn.benchmark for more optiSmized performance if the 
    # input size will not change at each iteration.
    torch.backends.cudnn.benchmark = True
    
    # Print the initial status of training.
    if is_base_rank:
        print('-'*80)
        print(f'Starting to train for {args.epochs} epochs and '
              f'batch size: {args.batch_size}.')
    
    # Start the epoch loop.
    ddp_utils.barrier(is_ddp)
    for i, epoch in enumerate(range(args.start_epoch, args.epochs)):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        if is_base_rank:
            print('-'*50, f'Starting epoch: {epoch}')
        
        # Train for one epoch.
        ddp_utils.barrier(is_ddp)
        train_outputs = train_one_epoch(train_loader, model, criterion, 
                                        optimizer, scheduler, epoch, args)
        
        # Do something with the train_outputs if needed.
        # ...

        # Validate for one epoch.
        ddp_utils.barrier(is_ddp)
        valid_outputs = validate_one_epoch(valid_loader, model, 
                                           criterion, epoch, args)
        
        # Do something with the valid_outputs if needed.
        # ...
        
        # Check for early stopping.
        ddp_utils.barrier(is_ddp)
        if is_base_rank:
            if early_stop_counter >= args.early_stop_patience:
                print('-'*50, 'Early stopping!')
                if not is_ddp:
                    break
                else:
                    store.set('early_stop', 'enabled')
        if is_ddp:
            if store.get('early_stop') == 'enabled':
                break
        
        # Adjust the learning rate in each epoch, if needed.
        # scheduelr.step()
        
        # Sync all the processes at the end of the epoch.
        ddp_utils.barrier(is_ddp)
        
    #---------------------------------------
    #-- End of training
    
    if is_base_rank:
        wandb.finish(quiet=True)
    ddp_utils.barrier(is_ddp)
    ddp_utils.cleanup_ddp(is_ddp)
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

#-------------------------------------------------------------------------------
# Training and validation loops
#-------------------------------------------------------------------------------

#---------------------------------------
# - F: train_one_epch

def train_one_epoch(train_loader: Iterable, 
                    model: torch.nn.Module, 
                    criterion: Callable, 
                    optimizer: torch.optim.Optimizer, 
                    scheduler: torch.optim.lr_scheduler,
                    epoch: int, 
                    args: argparse.Namespace) -> dict:
    """Train the model for one epoch.
    
    Args:
        train_loader (Iterable): Dataloader for training.
        model (torch.nn.Module): PyTorch model to be trained.
        criterion (Callable): loss function to be used for training
        optimizer (torch.optim.Optimizer): optimizer to be used for training.
        scheduler (torch.optim.lr_scheduler): scheduler to be used for training.
        epoch (int): the current epoch.
        args (argparse.Namespace): parsed arguments.
        
    Raises:
        ValueError: raises an error if the "log_freq" is not a positive 
            number.
    
    Returns:
        train_outputs (dict): dictionary containing the outputs of the training
            loop.
    """
    # Mention global variables that may be reassigned in this function.
    #...
    
    # Define the train_ouputs dictionary. This can be useful if you need to 
    # return anything from this function to the epoch loop. Do not use this 
    # dictionary for logging as logging will automatically be done using global 
    # dictionaries. Also, do not return global variables.
    train_outputs = dict()
    
    # Set up the validation loop.
    model.train()
    if is_base_rank:
        pbar = tqdm(total=len(train_loader), desc=f'Training', unit='batch')
    if args.log_freq == -1:
        train_log_freq = len(train_loader)
    elif args.log_freq > 0:
        train_log_freq = min(args.log_freq, len(train_loader))
    else:
        raise ValueError('The log_freq should be positive.')
        
    # Start the training loop.
    for i, batch in enumerate(train_loader):
        if args.gpu is None:
            inputs = batch['image'].cuda()
            labels = batch['label'].cuda()
        else:
            inputs = batch['image'].cuda(args.gpu)
            labels = batch['label'].cuda(args.gpu)
        
        # Forward pass + calculate the loss.
        optimizer.zero_grad()
        preds = model(inputs)
        train_loss = criterion(preds, labels)
    
        # Backward pass + optimization
        train_loss.backward()
        optimizer.step()
        
        # Adjust the learning rate in each step, if needed.
        # scheduler.step(epoch + i/len(train_loader))
        
        # Calculate the training metrics.
        train_acc = torchmetrics.functional.accuracy(preds.softmax(dim=-1), 
                                                     labels)
        
        # Update the train_outputs dictionary, if needed.
        # ...
        
        # Log the step-wise stats.
        if i>0 and (i+1) % train_log_freq == 0:
            collect_log(train_loss, 'train_loss', 's')   
            collect_log(train_acc, 'train_acc', 's')
            if is_base_rank:
                wandb.log({'train_step_loss': train_loss.item(),
                           'train_step_acc': train_acc.item(),
                           'step': epoch*len(train_loader) + i})
            
        # Update the progress bar every step.
        if is_base_rank:
            pbar.update(1)
            pbar.set_postfix_str(f'batch train loss: {train_loss.item():.2f}')
        
    # Do base-rank operations at the end of the training loop.
    if is_base_rank:

        # Log the epoch-wise stats.
        train_epoch_loss = statistics.mean(
            train_step_logs['train_loss'][-len(train_loader):])
        train_epoch_acc = statistics.mean(
            train_step_logs['train_acc'][-len(train_loader):])
        collect_log(train_epoch_loss, 'train_epoch_loss', 'e')
        collect_log(train_epoch_acc, 'train_epoch_acc', 'e')
        wandb.log({'train_epoch_loss': train_epoch_loss,
                   'train_epoch_acc': train_epoch_acc,
                   'epoch': epoch})
    
        # Close the progress bar.
        pbar.close()
        time.sleep(0.1)
        print(f"The average train loss for epoch {epoch}: ", 
              f"{train_epoch_loss:.2f}")
        print(f"The average train accuracy for epoch {epoch}: ", 
              f"{train_epoch_acc:.2f}")     
        print('-'*20)
        
    return train_outputs

#---------------------------------------
# - F: validate_one_epch

def validate_one_epoch(valid_loader: Iterable, 
                       model: torch.nn.Module, 
                       criterion: Callable, 
                       epoch: int,
                       args: argparse.Namespace) -> dict:
    """Validate the model for one epoch.
    
    Args:
        valid_loader (Iterable): Dataloader for validation.
        model (torch.nn.Module): PyTorch model to be trained.
        criterion (Callable): loss function to be used for training.
        epoch (int): the current epoch.
        args (argparse.Namespace): parsed arguments.
        
    Raises:
        ValueError: raises an error if the "log_freq" is not a positive 
            number.
            
    Returns:
        valid_outputs (dict): dictionary containing the outputs of the
            validation loop.
    """
    # Mention global variables that may be reassigned in this function.
    global best_benchmark
    global early_stop_counter
    
    # Define the valid_outputs dictionary. This can be useful if you need to 
    # return anythinng from this function to the epoch loop. Do not use this 
    # dictionary for logging as logging will automatically be done using global 
    # dictionaries. Also, do not return global variables.
    valid_outputs = dict()
    
    # Set up the validation loop.
    model.eval()
    if is_base_rank:
        pbar = tqdm(total=len(valid_loader), desc=f'Validation', 
                    unit='batch')  
    if args.log_freq == -1:
        valid_log_freq = len(valid_loader)
    elif args.log_freq > 0:
        valid_log_freq = min(args.log_freq, len(valid_loader))
    else:
        raise ValueError('The log_freq should be positive.')
        
    # Start the validation loop.
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            if args.gpu is None:
                inputs = batch['image'].cuda()
                labels = batch['label'].cuda() 
            else:
                inputs = batch['image'].cuda(args.gpu)
                labels = batch['label'].cuda(args.gpu)

            # Forward pass + calculate the loss
            preds = model(inputs)
            valid_loss = criterion(preds, labels)
            
            # Calculate the validation metrics.
            valid_acc = torchmetrics.functional.accuracy(preds.softmax(dim=-1), 
                                                         labels)
            
            # Update the valid_outputs dictionary, if needed.
            ...
    
            # Log the step-wise stats.
            if  i>0 and (i+1) % valid_log_freq == 0:
                collect_log(valid_loss, 'valid_loss', 's')
                collect_log(valid_acc, 'valid_acc', 's')
                if is_base_rank:
                    wandb.log({'valid_step_loss': valid_loss.item(),
                            'valid_step_acc': valid_acc.item(),
                            'step': epoch*len(valid_loader) + i})
                
            # Update the progress bar every step.
            if is_base_rank:
                pbar.update(1)
                pbar.set_postfix_str(f'batch valid loss: '
                                     f'{valid_loss.item():.2f}')
        
    # Do base-rank operations at the end of the validation loop.
    if is_base_rank:
        
        # Log the epochwise stats.
        valid_epoch_loss = statistics.mean(
            valid_step_logs['valid_loss'][-len(valid_loader):])
        valid_epoch_acc = statistics.mean(
            valid_step_logs['valid_acc'][-len(valid_loader):])
        collect_log(valid_epoch_loss, 'valid_epoch_loss', 'e')
        collect_log(valid_epoch_acc, 'valid_epoch_acc', 'e')
        wandb.log({'valid_epoch_loss': valid_epoch_loss,
                   'valid_epoch_acc': valid_epoch_acc,
                   'epoch': epoch})
    
        # Close the progress bar.
        pbar.close() 
        time.sleep(0.1)
        print(f"The average valid loss for epoch {epoch}: ", 
              f"{valid_epoch_loss:.2f}")   
        print(f"The average valid accuracy for epoch {epoch}: ", 
              f"{valid_epoch_acc:.2f}")
        
        # Save the best model if the current benchmark is better than the 
        # already measured best bechmark. Else, increment the early_stop_counter.
        if best_benchmark is None: 
            best_benchmark = valid_epoch_loss
        if valid_epoch_loss <= best_benchmark:
            best_benchmark = valid_epoch_loss
            if args.parallel_mode in ['dp', 'ddp']:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            pytorch_utils.save_checkpoint(
                {'state_dict': state_dict,'epoch': epoch, 'step':i}, 
                checkpoint_dir = checkpoint_dir,
                add_text = f'val-loss={best_benchmark:.2f}')
            early_stop_counter = 0
            print('-'*20)
            print(f'A new best model is saved at epoch {epoch}.')
            print(f'The best new valid_epoch_loss is: {best_benchmark:.2f}')
        else:
            early_stop_counter += 1
            
    return valid_outputs
    
#-------------------------------------------------------------------------------
# Helper functions
# These functions work with the global variables in this script, and hence, 
# cannot be defined in other scripts.
#-------------------------------------------------------------------------------

#---------------------------------------
# - F: collect_log

def collect_log(log_key: Union[torch.tensor, float], 
                log_key_name: str, mode: str,
                return_gathered: bool = False) -> float:
    """Collects the values for a tensor from all the ranks and appends that
    collected value to either the training or the validation logs. Collecting
    could be done across steps or across epochs.

    Args:
        log_key (Union[torch.tensor, float]): the key tensor or value to 
            be logged.
        log_key_name (str): the name of the tensor to be logged.
        mode (str): the mode of the log. Either 's' (step) or 'e' (epoch).
        return_gathered (bool): whether to return the gathered tensor. 
            Defaults to False.
            
    Raises:
        ValueError: if the log_key_name does not include "train" or "valid.
    
    Returns:
        gathered_log_key (float): The float value for the gathered tensor.
            Defaults to False.
    """
    # Determining the mode and the logs.
    assert mode in ['s', 'e'], \
        "The mode should be either 's' (step) or 'e' (epoch)."   
    if mode == 's':
        if 'train' in log_key_name:
            logs = train_step_logs
        elif 'valid' in log_key_name:
            logs = valid_step_logs
        else:
            raise ValueError('Unknown log_key_name!')
    else:
        if 'train' in log_key_name:
            logs = train_epoch_logs
        elif 'valid' in log_key_name:
            logs = valid_epoch_logs
        else:
            raise ValueError('Unknown log_key_name!')
    
    # Gathering the log_key. If the log key is a float, there is no need to 
    # gathering. 
    if type(log_key) == torch.Tensor:
        gathered_log_key = ddp_utils.gather_tensor(log_key, is_ddp = is_ddp,
                                        is_base_rank = is_base_rank)
    else:
        gathered_log_key = log_key
    
    # Updating the log lists.
    if is_base_rank:
        log_list = logs.get(log_key_name, [])
        log_list.append(gathered_log_key)
        logs[log_key_name] = log_list
    
    # Return the gathered log_key. 
    if return_gathered:
        return gathered_log_key

#-------------------------------------------------------------------------------
# Run
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()