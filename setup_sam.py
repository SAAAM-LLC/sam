#!/usr/bin/env python3
"""
setup_sam.py - SAM Environment Setup and Dataset Processing

This script sets up the SAM environment and processes various dataset formats
for training or fine-tuning the Synergistic Autonomous Machine.

Usage:
    python setup_sam.py --dataset path/to/data --format jsonl --output ./data
    python setup_sam.py --create-dirs-only  # Just create directory structure
"""

import os
import json
import csv
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
import pandas as pd
from tqdm import tqdm
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SAMSetup:
    """SAM environment setup and dataset processing"""
    
    def __init__(self, base_dir: str = "./sam_workspace"):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        self.configs_dir = self.base_dir / "configs"
        
        # Dataset processing stats
        self.processed_examples = 0
        self.skipped_examples = 0
        self.error_examples = 0
        
    def create_directory_structure(self):
        """Create the SAM workspace directory structure"""
        directories = [
            self.base_dir,
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "vocabulary",
            self.models_dir,
            self.models_dir / "checkpoints",
            self.models_dir / "pretrained",
            self.logs_dir,
            self.logs_dir / "training",
            self.logs_dir / "evolution",
            self.logs_dir / "dreams",
            self.configs_dir,
            self.base_dir / "experiments",
            self.base_dir / "exports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Create default config files
        self._create_default_configs()
        
        # Create gitignore
        self._create_gitignore()
        
        logger.info(f"SAM workspace created at: {self.base_dir}")
    
    def _create_default_configs(self):
        """Create default configuration files"""
        
        # Training config
        train_config = {
            "model": {
                "initial_hidden_dim": 768,
                "initial_num_layers": 6,
                "max_hidden_dim": 2048,
                "max_num_layers": 12,
                "growth_factor": 1.3,
                "neurochemical_enabled": True,
                "biological_computing": True,
                "emergent_representations": True,
                "multi_level_evolution": True
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 3e-5,
                "warmup_steps": 1000,
                "max_steps": 100000,
                "eval_steps": 1000,
                "save_steps": 5000,
                "gradient_clip": 1.0,
                "mixed_precision": True
            },
            "data": {
                "max_sequence_length": 2048,
                "train_split": 0.9,
                "eval_split": 0.1,
                "shuffle": True,
                "num_workers": 4
            },
            "evolution": {
                "evolve_every": 1000,
                "dream_cycle_minutes": 0.2,
                "consciousness_check_steps": 100
            }
        }
        
        with open(self.configs_dir / "train_config.json", "w") as f:
            json.dump(train_config, f, indent=2)
        
        # Interaction config
        interact_config = {
            "generation": {
                "max_length": 200,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            },
            "interface": {
                "save_conversations": True,
                "conversation_history": 20,
                "auto_evolve": True,
                "dream_during_idle": True
            },
            "privacy": {
                "enable_private_context": True,
                "log_interactions": False,
                "anonymize_data": True
            }
        }
        
        with open(self.configs_dir / "interact_config.json", "w") as f:
            json.dump(interact_config, f, indent=2)
        
        # Hardware config
        hardware_config = {
            "gpu_memory_threshold": 0.8,
            "cpu_fallback": True,
            "mixed_precision": True,
            "gradient_checkpointing": False,
            "model_parallel": False,
            "offload_optimizer": False
        }
        
        with open(self.configs_dir / "hardware_config.json", "w") as f:
            json.dump(hardware_config, f, indent=2)
        
        logger.info("Created default configuration files")
    
    def _create_gitignore(self):
        """Create .gitignore file"""
        gitignore_content = """
# SAM workspace artifacts
*.pyc
__pycache__/
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/

# SAM specific
/data/raw/
/data/processed/
/models/checkpoints/
/logs/
*.pt
*.pth
*.safetensors

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
"""
        
        with open(self.base_dir / ".gitignore", "w") as f:
            f.write(gitignore_content.strip())
    
    def process_dataset(self, dataset_path: str, format_type: str, 
                       output_dir: str = None, **kwargs):
        """Process dataset in various formats"""
        
        if output_dir is None:
            output_dir = self.data_dir / "processed"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_path = Path(dataset_path)
        
        # Determine format if not specified
        if format_type == "auto":
            format_type = self._detect_format(dataset_path)
        
        logger.info(f"Processing {format_type} dataset: {dataset_path}")
        
        # Process based on format
        processors = {
            "jsonl": self._process_jsonl,
            "json": self._process_json,
            "txt": self._process_txt,
            "csv": self._process_csv,
            "parquet": self._process_parquet,
            "huggingface": self._process_huggingface
        }
        
        if format_type not in processors:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Reset stats
        self.processed_examples = 0
        self.skipped_examples = 0
        self.error_examples = 0
        
        # Process the dataset
        output_file = output_dir / f"processed_{dataset_path.stem}.jsonl"
        processors[format_type](dataset_path, output_file, **kwargs)
        
        # Create dataset info
        self._create_dataset_info(output_file, dataset_path, format_type)
        
        logger.info(f"Dataset processing complete:")
        logger.info(f"  Processed: {self.processed_examples}")
        logger.info(f"  Skipped: {self.skipped_examples}")
        logger.info(f"  Errors: {self.error_examples}")
        logger.info(f"  Output: {output_file}")
        
        return output_file
    
    def _detect_format(self, file_path: Path) -> str:
        """Auto-detect dataset format"""
        suffix = file_path.suffix.lower()
        
        format_map = {
            ".jsonl": "jsonl",
            ".json": "json",
            ".txt": "txt",
            ".csv": "csv",
            ".parquet": "parquet"
        }
        
        return format_map.get(suffix, "txt")
    
    def _process_jsonl(self, input_path: Path, output_path: Path, 
                      text_field: str = "text", **kwargs):
        """Process JSONL format"""
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(tqdm(infile, desc="Processing JSONL")):
                try:
                    data = json.loads(line.strip())
                    processed = self._extract_text(data, text_field)
                    
                    if processed:
                        sam_example = {
                            "text": processed["text"],
                            "metadata": {
                                "source": str(input_path),
                                "line_number": line_num,
                                "original_keys": list(data.keys()),
                                **processed.get("metadata", {})
                            }
                        }
                        
                        outfile.write(json.dumps(sam_example, ensure_ascii=False) + "\n")
                        self.processed_examples += 1
                    else:
                        self.skipped_examples += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    self.error_examples += 1
    
    def _process_json(self, input_path: Path, output_path: Path, 
                     text_field: str = "text", **kwargs):
        """Process JSON format"""
        with open(input_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "data" in data:
            items = data["data"]
        elif isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
            # Find the list field
            for key, value in data.items():
                if isinstance(value, list):
                    items = value
                    break
        else:
            items = [data]
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for idx, item in enumerate(tqdm(items, desc="Processing JSON")):
                try:
                    processed = self._extract_text(item, text_field)
                    
                    if processed:
                        sam_example = {
                            "text": processed["text"],
                            "metadata": {
                                "source": str(input_path),
                                "index": idx,
                                **processed.get("metadata", {})
                            }
                        }
                        
                        outfile.write(json.dumps(sam_example, ensure_ascii=False) + "\n")
                        self.processed_examples += 1
                    else:
                        self.skipped_examples += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing item {idx}: {e}")
                    self.error_examples += 1
    
    def _process_txt(self, input_path: Path, output_path: Path, 
                    chunk_size: int = 1000, overlap: int = 100, **kwargs):
        """Process plain text format"""
        with open(input_path, 'r', encoding='utf-8') as infile:
            text = infile.read()
        
        # Split into chunks
        chunks = self._split_text(text, chunk_size, overlap)
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for idx, chunk in enumerate(tqdm(chunks, desc="Processing TXT")):
                try:
                    if len(chunk.strip()) > 50:  # Minimum chunk size
                        sam_example = {
                            "text": chunk.strip(),
                            "metadata": {
                                "source": str(input_path),
                                "chunk_index": idx,
                                "chunk_size": len(chunk)
                            }
                        }
                        
                        outfile.write(json.dumps(sam_example, ensure_ascii=False) + "\n")
                        self.processed_examples += 1
                    else:
                        self.skipped_examples += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing chunk {idx}: {e}")
                    self.error_examples += 1
    
    def _process_csv(self, input_path: Path, output_path: Path, 
                    text_field: str = None, **kwargs):
        """Process CSV format"""
        df = pd.read_csv(input_path)
        
        # Auto-detect text field if not specified
        if text_field is None:
            text_candidates = ['text', 'content', 'message', 'body', 'description']
            for candidate in text_candidates:
                if candidate in df.columns:
                    text_field = candidate
                    break
            
            if text_field is None:
                # Use the column with longest average text
                text_field = df.select_dtypes(include=['object']).apply(
                    lambda x: x.astype(str).str.len().mean()
                ).idxmax()
        
        logger.info(f"Using text field: {text_field}")
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV"):
                try:
                    text_content = str(row[text_field]).strip()
                    
                    if len(text_content) > 10:  # Minimum text length
                        metadata = {col: str(val) for col, val in row.items() 
                                   if col != text_field}
                        
                        sam_example = {
                            "text": text_content,
                            "metadata": {
                                "source": str(input_path),
                                "row_index": idx,
                                "text_field": text_field,
                                **metadata
                            }
                        }
                        
                        outfile.write(json.dumps(sam_example, ensure_ascii=False) + "\n")
                        self.processed_examples += 1
                    else:
                        self.skipped_examples += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    self.error_examples += 1
    
    def _process_parquet(self, input_path: Path, output_path: Path, 
                        text_field: str = None, **kwargs):
        """Process Parquet format"""
        df = pd.read_parquet(input_path)
        
        # Convert to CSV processing
        temp_csv = input_path.with_suffix('.temp.csv')
        df.to_csv(temp_csv, index=False)
        
        try:
            self._process_csv(temp_csv, output_path, text_field, **kwargs)
        finally:
            temp_csv.unlink(missing_ok=True)
    
    def _process_huggingface(self, dataset_name: str, output_path: Path, 
                           split: str = "train", **kwargs):
        """Process HuggingFace dataset"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        dataset = load_dataset(dataset_name, split=split)
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for idx, example in enumerate(tqdm(dataset, desc="Processing HF Dataset")):
                try:
                    # Try common text fields
                    text_content = None
                    for field in ['text', 'content', 'input', 'prompt']:
                        if field in example:
                            text_content = example[field]
                            break
                    
                    if text_content and len(str(text_content).strip()) > 10:
                        sam_example = {
                            "text": str(text_content).strip(),
                            "metadata": {
                                "source": dataset_name,
                                "split": split,
                                "index": idx,
                                **{k: str(v) for k, v in example.items() 
                                   if k != 'text' and not isinstance(v, (list, dict))}
                            }
                        }
                        
                        outfile.write(json.dumps(sam_example, ensure_ascii=False) + "\n")
                        self.processed_examples += 1
                    else:
                        self.skipped_examples += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing example {idx}: {e}")
                    self.error_examples += 1
    
    def _extract_text(self, data: Dict, text_field: str) -> Dict[str, Any]:
        """Extract text content from data item"""
        if isinstance(data, str):
            return {"text": data}
        
        if not isinstance(data, dict):
            return None
        
        # Try specified field first
        if text_field in data and data[text_field]:
            return {
                "text": str(data[text_field]),
                "metadata": {k: v for k, v in data.items() if k != text_field}
            }
        
        # Try common field names
        common_fields = ['text', 'content', 'message', 'body', 'input', 'prompt']
        for field in common_fields:
            if field in data and data[field]:
                return {
                    "text": str(data[field]),
                    "metadata": {k: v for k, v in data.items() if k != field}
                }
        
        # If conversation format, combine turns
        if 'conversations' in data or 'messages' in data:
            conversations = data.get('conversations', data.get('messages', []))
            text_parts = []
            for turn in conversations:
                if isinstance(turn, dict):
                    role = turn.get('role', turn.get('from', 'user'))
                    content = turn.get('content', turn.get('value', ''))
                    text_parts.append(f"{role}: {content}")
            
            if text_parts:
                return {
                    "text": "\n".join(text_parts),
                    "metadata": {k: v for k, v in data.items() 
                               if k not in ['conversations', 'messages']}
                }
        
        return None
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
                
            start = end - overlap
        
        return chunks
    
    def _create_dataset_info(self, output_file: Path, source_path: Path, 
                           format_type: str):
        """Create dataset information file"""
        info = {
            "source_file": str(source_path),
            "format": format_type,
            "processed_file": str(output_file),
            "processing_stats": {
                "processed_examples": self.processed_examples,
                "skipped_examples": self.skipped_examples,
                "error_examples": self.error_examples,
                "total_attempted": self.processed_examples + self.skipped_examples + self.error_examples
            },
            "file_stats": {
                "size_bytes": output_file.stat().st_size,
                "checksum": self._calculate_checksum(output_file)
            }
        }
        
        info_file = output_file.with_suffix('.info.json')
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def create_vocabulary_file(self, processed_files: List[str], 
                             output_file: str = None, min_freq: int = 5):
        """Create vocabulary file from processed datasets"""
        if output_file is None:
            output_file = self.data_dir / "vocabulary" / "vocab.txt"
        
        vocab_counter = {}
        
        for file_path in processed_files:
            logger.info(f"Processing vocabulary from: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Building vocabulary"):
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')
                        
                        # Simple tokenization for vocabulary
                        tokens = text.lower().split()
                        for token in tokens:
                            # Clean token
                            clean_token = ''.join(c for c in token if c.isalnum() or c in '-_')
                            if len(clean_token) > 1:
                                vocab_counter[clean_token] = vocab_counter.get(clean_token, 0) + 1
                    
                    except Exception as e:
                        continue
        
        # Filter by frequency and save
        filtered_vocab = {word: count for word, count in vocab_counter.items() 
                         if count >= min_freq}
        
        # Sort by frequency
        sorted_vocab = sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for word, count in sorted_vocab:
                f.write(f"{word}\n")
        
        logger.info(f"Vocabulary saved: {len(sorted_vocab)} words to {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Setup SAM environment and process datasets")
    
    # Setup arguments
    parser.add_argument("--workspace", type=str, default="./sam_workspace",
                       help="SAM workspace directory")
    parser.add_argument("--create-dirs-only", action="store_true",
                       help="Only create directory structure")
    
    # Dataset processing arguments
    parser.add_argument("--dataset", type=str, help="Path to dataset file")
    parser.add_argument("--format", type=str, default="auto",
                       choices=["auto", "jsonl", "json", "txt", "csv", "parquet", "huggingface"],
                       help="Dataset format")
    parser.add_argument("--output", type=str, help="Output directory for processed data")
    parser.add_argument("--text-field", type=str, help="Field name containing text")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Chunk size for text files")
    parser.add_argument("--overlap", type=int, default=100,
                       help="Overlap size for text chunks")
    
    # Vocabulary arguments
    parser.add_argument("--create-vocab", action="store_true",
                       help="Create vocabulary file from processed data")
    parser.add_argument("--vocab-min-freq", type=int, default=5,
                       help="Minimum frequency for vocabulary")
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = SAMSetup(args.workspace)
    
    # Create directories
    setup.create_directory_structure()
    
    if args.create_dirs_only:
        logger.info("Directory structure created. Exiting.")
        return
    
    # Process dataset if provided
    if args.dataset:
        processed_file = setup.process_dataset(
            dataset_path=args.dataset,
            format_type=args.format,
            output_dir=args.output,
            text_field=args.text_field,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
        
        # Create vocabulary if requested
        if args.create_vocab:
            setup.create_vocabulary_file(
                [str(processed_file)],
                min_freq=args.vocab_min_freq
            )
    
    logger.info("SAM setup complete!")

if __name__ == "__main__":
    main()
