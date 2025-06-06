# data_loader.py - SAM Data Loading Components
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SAMDataset(Dataset):
    """Dataset for SAM training"""
    
    def __init__(self, data_files: List[str], max_length: int = 2048, model=None):
        self.data_files = data_files
        self.max_length = max_length
        self.model = model
        self.examples = []
        
        # Load all examples
        self._load_examples()
    
    def _load_examples(self):
        """Load examples from data files"""
        for file_path in self.data_files:
            logger.info(f"Loading data from: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        example = json.loads(line.strip())
                        if 'text' in example:
                            self.examples.append(example)
                    except Exception as e:
                        logger.warning(f"Skipping line {line_num} in {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        
        # Process text through SAM's segmentation if model provided
        if self.model:
            try:
                concept_ids, _ = self.model.process_text(text)
                
                # Truncate if too long
                if len(concept_ids) > self.max_length:
                    concept_ids = concept_ids[:self.max_length]
                
                return {
                    'input_ids': concept_ids,
                    'labels': concept_ids,  # For language modeling
                    'metadata': example.get('metadata', {})
                }
            except Exception as e:
                logger.warning(f"Error processing example {idx}: {e}")
                # Fallback to character-level processing
                chars = [ord(c) % 256 for c in text[:self.max_length]]
                return {
                    'input_ids': chars,
                    'labels': chars,
                    'metadata': example.get('metadata', {})
                }
        else:
            # Simple character-level fallback
            chars = [ord(c) % 256 for c in text[:self.max_length]]
            return {
                'input_ids': chars,
                'labels': chars,
                'metadata': example.get('metadata', {})
            }

class SAMDataLoader:
    """Enhanced data loader for SAM"""
    
    def __init__(self, dataset, batch_size=4, shuffle=True, num_workers=4, distributed=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.distributed = distributed
        
        # Setup sampler
        if distributed:
            self.sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            self.sampler = None
        
        # Create data loader
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and not distributed,
            sampler=self.sampler,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available()
        )
    
    def _collate_fn(self, batch):
        """Custom collation function for SAM data"""
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        metadata = [item['metadata'] for item in batch]
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for seq_input, seq_labels in zip(input_ids, labels):
            # Pad input
            pad_length = max_len - len(seq_input)
            padded_input = seq_input + [0] * pad_length
            padded_label = seq_labels + [-100] * pad_length  # -100 is ignored in loss
            
            # Create attention mask
            attention_mask = [1] * len(seq_input) + [0] * pad_length
            
            padded_input_ids.append(padded_input)
            padded_labels.append(padded_label)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.float),
            'metadata': metadata
        }
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)
