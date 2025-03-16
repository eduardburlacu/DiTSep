import torch
import os
import logging
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .wsj0_mix import max_collator
log = logging.getLogger(__name__)

class WSJ0LatentDataset(Dataset):
    """
    Dataset for cached latent samples from WSJ0 mixtures.
    """
    
    def __init__(self, latent_dir, wsj0_dataset, device=None):
        """
        Initialize the WSJ0 latent dataset.
        
        Args:
            latent_dir: Directory containing latent files and metadata
            wsj0_dataset: Original WSJ0_mix dataset for target retrieval
            device: Device to place tensors on (None keeps them on CPU)
        """
        self.latent_dir = latent_dir
        self.wsj0_dataset = wsj0_dataset
        self.device = device
        
        # Load metadata
        metadata_path = os.path.join(latent_dir, "metadata.pt")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
            
        self.metadata = torch.load(metadata_path)
        log.info(f"Loaded WSJ0LatentDataset with {self.metadata['total_samples']} samples")
        
        # Cache for recently used latents
        self.cache = {}
        self.max_cache_size = 100
        
    def __len__(self):
        return self.metadata['total_samples']
    
    def _load_latent(self, idx):
        """Load a latent from file or cache"""
        if idx in self.cache:
            return self.cache[idx]
            
        # Load latent file
        latent_path = os.path.join(self.latent_dir, f"latent_{idx:06d}.pt")
        latent_data = torch.load(latent_path)
        
        # Update cache
        if len(self.cache) >= self.max_cache_size:
            # Remove a random entry
            self.cache.pop(next(iter(self.cache)))
        self.cache[idx] = latent_data
        
        return latent_data
    
    def __getitem__(self, idx):
        """
        Get a sample by index.
        
        Returns:
            target: Target sources from original dataset
            latent: Generated latent
        """
        # Load the latent data
        latent_data = self._load_latent(idx)
        dataset_idx = self.metadata['sample_indices'][idx]
        
        # Get target from original dataset
        _, target = self.wsj0_dataset[dataset_idx]
        
        # Get latent
        latent = latent_data['latent']
        
        # Return target and latent (device transfer happens in collate_fn)
        return target, latent
    
def create_latent_dataloader(latent_dir, wsj0_dataset, batch_size=16, shuffle=True, num_workers=4):
    """
    Create a DataLoader for the WSJ0 latent dataset.
    
    Args:
        latent_dir: Directory containing latent files and metadata
        wsj0_dataset: Original WSJ0_mix dataset instance
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes
        
    Returns:
        DataLoader for the latent dataset
    """

    
    latent_dataset = WSJ0LatentDataset(latent_dir, wsj0_dataset)
    
    return DataLoader(
        latent_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

# Update the max_collator function to handle device placement
def device_aware_max_collator(batch, device=None):
    """
    Collate a batch of samples and optionally move them to the specified device.
    """
    collated_batch = max_collator(batch)
    
    if device is not None:
        # Move tensors to device if requested
        if isinstance(collated_batch, tuple):
            collated_batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t 
                                  for t in collated_batch)
    
    return collated_batch