#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

# Standard built-in modules
import argparse
import gc
import os
import warnings
from typing import Callable, Union

# Third-party modules
import torch

#-------------------------------------------------------------------------------
# Configurations
#-------------------------------------------------------------------------------

warnings.filterwarnings("ignore") 

#-------------------------------------------------------------------------------
# General tools
#-------------------------------------------------------------------------------

#-----------------------------------
# - F: setup_ddp

def setup_ddp(gpu: int, args: argparse.Namespace):
    """Prepare the PyTorch Distributed Data Parallel pipeline.
    This function should be called in the beginning of the train() function.
    Please note: 
        - Almost always one process is run on one GPU, so the terms 
          "GPU", "process", and "device" all denote a single GPU.
        - Each "node" is an independant computer on the network that can have
          different number of GPUs available.
        - By default, all nodes should have similiar number of GPUs available.
    
    Args:
        gpu (int): the rank of the current gpu within the current node.
            args (argparse.Namespace): parsed arguments.
    """
    # Add the IP address and port of the main node to the environment variables.
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port  
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ["NODE_RANK"] = str(args.node_rank)
    
    # Calculate the total rank of the current GPU.
    # gpu is the rank of the current gpu within the current node.
    # gpu_rank is the rank of the current gpu with respect to all available 
    # nodes.
    args.gpu = gpu
    args.gpu_rank = args.node_rank * args.ngpus_per_node + args.gpu
    
    # Initialize the DDP process_group. 
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.init_process_group(backend=args.dist_backend, 
                            init_method=args.dist_url,
                            world_size=args.world_size, 
                            rank=args.gpu_rank)
    
    # Log the status.
    if args.gpu is not None and args.verbose_ddp:
        print(f"Set up DDP on: node rank -> {args.node_rank}, ", end = "")
        print(f"GPU rank -> {args.gpu}, ", end = "")
        print(f"and total rank -> {args.gpu_rank}")

#---------------------------------------
# - F: cleanup_ddp

def cleanup_ddp(is_ddp: bool = True):
    """End the PyTorch Distributed Data Parallel.
    
    Args:
        is_ddp (bool): whether or not the training is running in DDP. 
            Defaults to True.
    """
    if is_ddp:
        torch.distributed.destroy_process_group()

#---------------------------------------
# - F: barrier
        
def barrier(is_ddp: bool = True):
    """Put a process barrier checkpoint if training is in DDP.
    
    Args:
        is_ddp (bool): whether or not the training is running in DDP. 
            Defaults to True.
    """
    if is_ddp:
        torch.distributed.barrier()
        
#-------------------------------------------------------------------------------
# Collection and distribution tools
#-------------------------------------------------------------------------------

#---------------------------------------
# - F: gather_tensor
        
def gather_tensor(t: torch.Tensor, 
                  reduce_fn: Callable = torch.mean,
                  is_ddp: bool = False,
                  is_base_rank: bool = True) -> Union[torch.Tensor, 
                                                      list, float]:
    """Gathers a list of tensors across all GPUs (if in DDP). 
    Can also reduce the list in one summary measure like mean. This could
    also be done using. torch.distributed.reduce()
    
    Args:
        t (torch.tensor): tensor to be gathered.
        reduce_fn (Callable): a torch function to be used for reducing the 
            tensor. Defaults to torch.mean.
        is_ddp (bool): whether or not the training is running in DDP. 
            Defaults to False.
        is_base_rank (bool): whether or not the rank is 0 (or base). 
            Defaults to True.
    
    Returns:
        output (Union[torch.Tensor, list, float]): a gathered list of the target 
            tensor or a reduced value of that.
    """
    with torch.no_grad():
        if not is_ddp:
            return t.detach().item()
        output = [torch.zeros_like(t) 
                  for _ in range(int(os.environ['WORLD_SIZE']))]
        torch.distributed.gather(t.detach(), 
                                 output if is_base_rank else None, dst=0)
        if is_base_rank:
            if reduce_fn is not None:
                output = reduce_fn(torch.tensor(output)).item()
            return output
        
#---------------------------------------
# - C: DistributedProxySampler

class DistributedProxySampler(torch.utils.data.distributed.DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Input sampler is assumed to be of constant size.
    """

    def __init__(self, sampler: torch.utils.data.sampler.WeightedRandomSampler, 
                 num_replicas: int = None, rank: int = None):        
        super(DistributedProxySampler, self).__init__(sampler, 
                                                      num_replicas=num_replicas, 
                                                      rank=rank, shuffle=False)
        """
        Args:
            sampler (torch.utils.data.sampler.WeightedRandomSampler): 
                input data sampler.
            num_replicas (int, optional): number of processes participating in
                distributed training.
            rank (iint, optional): rank of the current process within 
                num_replicas.
        """
        self.sampler = sampler

    def __iter__(self):
        """
        Raises:
            RuntimeError: if the length of sampler data differs from the 
                dataset size
        
        Yields: 
            indices for a batch feeding to a single process (GPU).
        """
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError(f"{len(indices)} vs {self.num_samples}")
        
        return iter(indices)