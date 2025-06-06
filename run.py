#!/usr/bin/env python3
"""
run.py - SAM Training and Interaction Interface

Main entry point for training and interacting with SAM models.

Usage:
    # Training
    python run.py --mode train --config configs/train_config.json --data data/processed/

    # Interactive mode
    python run.py --mode interact --model models/checkpoints/sam_latest

    # Resume training
    python run.py --mode train --resume models/checkpoints/sam_step_50000

    # Distributed training
    python run.py --mode train --distributed --world-size 4 --rank 0
"""

import os
import sys
import json
import argparse
import logging
import signal
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import wandb

# Add SAM to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sam import SAM, SAMConfig, create_sam_model
from data_loader import SAMDataset, SAMDataLoader
from trainer import SAMTrainer
from interactive import SAMInteractive

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sam_run.log')
    ]
)
logger = logging.getLogger(__name__)

class SAMRunner:
    """Main SAM runner for training and interaction"""
    
    def __init__(self, args):
        self.args = args
        self.config = None
        self.model = None
        self.trainer = None
        self.interactive = None
        
        # Setup signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.shutdown_requested = False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        
        if self.trainer:
            self.trainer.request_shutdown()
        
        if self.interactive:
            self.interactive.request_shutdown()
    
    def run(self):
        """Main run method"""
        try:
            if self.args.mode == "train":
                return self._run_training()
            elif self.args.mode == "interact":
                return self._run_interactive()
            elif self.args.mode == "evaluate":
                return self._run_evaluation()
            elif self.args.mode == "export":
                return self._run_export()
            else:
                raise ValueError(f"Unknown mode: {self.args.mode}")
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            return 1
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            logger.error(traceback.format_exc())
            return 1
    
    def _run_training(self):
        """Run training mode"""
        logger.info("Starting SAM training...")
        
        # Load configuration
        if self.args.config:
            with open(self.args.config, 'r') as f:
                config_dict = json.load(f)
        else:
            config_dict = {}
        
        # Apply command line overrides
        if self.args.batch_size:
            config_dict.setdefault("training", {})["batch_size"] = self.args.batch_size
        if self.args.learning_rate:
            config_dict.setdefault("training", {})["learning_rate"] = self.args.learning_rate
        if self.args.max_steps:
            config_dict.setdefault("training", {})["max_steps"] = self.args.max_steps
        
        # Setup distributed training if requested
        if self.args.distributed:
            self._setup_distributed()
        
        # Create or load model
        if self.args.resume:
            logger.info(f"Resuming training from: {self.args.resume}")
            self.model = SAM.load(self.args.resume, config_overrides=config_dict)
            self.config = self.model.config
        else:
            # Merge with model config
            model_config = config_dict.get("model", {})
            self.config = SAMConfig(**model_config)
            self.model, _ = create_sam_model(model_config)
        
        # Setup data loading
        data_loader = self._setup_data_loader(config_dict.get("data", {}))
        
        # Setup trainer
        training_config = config_dict.get("training", {})
        self.trainer = SAMTrainer(
            model=self.model,
            config=training_config,
            data_loader=data_loader,
            distributed=self.args.distributed,
            local_rank=getattr(self.args, 'local_rank', 0)
        )
        
        # Setup logging and monitoring
        if self.args.wandb and not self.args.distributed or (self.args.distributed and self.args.local_rank == 0):
            wandb.init(
                project="sam-training",
                config=config_dict,
                name=f"sam_run_{int(time.time())}"
            )
        
        # Train the model
        try:
            self.trainer.train()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.trainer.save_checkpoint("interrupted")
        
        # Cleanup distributed
        if self.args.distributed:
            dist.destroy_process_group()
        
        logger.info("Training completed")
        return 0
    
    def _run_interactive(self):
        """Run interactive mode"""
        logger.info("Starting SAM interactive mode...")
        
        # Load configuration
        config_path = self.args.interact_config or "configs/interact_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                interact_config = json.load(f)
        else:
            interact_config = {}
        
        # Load model
        if self.args.model:
            logger.info(f"Loading model from: {self.args.model}")
            self.model = SAM.load(self.args.model)
        else:
            # Create default model
            logger.info("Creating new SAM model")
            self.model, _ = create_sam_model()
        
        # Setup interactive interface
        self.interactive = SAMInteractive(
            model=self.model,
            config=interact_config,
            save_conversations=not self.args.no_save
        )
        
        # Run interactive session
        if self.args.web_interface:
            return self.interactive.run_web_interface(
                host=self.args.host,
                port=self.args.port
            )
        else:
            return self.interactive.run_console_interface()
    
    def _run_evaluation(self):
        """Run evaluation mode"""
        logger.info("Starting SAM evaluation...")
        
        # Load model
        if not self.args.model:
            raise ValueError("Model path required for evaluation")
        
        self.model = SAM.load(self.args.model)
        
        # Load evaluation data
        eval_data_path = self.args.eval_data or self.args.data
        if not eval_data_path or not os.path.exists(eval_data_path):
            raise ValueError("Evaluation data path required")
        
        # Run evaluation
        from evaluator import SAMEvaluator
        
        evaluator = SAMEvaluator(self.model)
        results = evaluator.evaluate(eval_data_path)
        
        # Save results
        output_path = self.args.output or f"eval_results_{int(time.time())}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to: {output_path}")
        return 0
    
    def _run_export(self):
        """Run export mode"""
        logger.info("Starting SAM model export...")
        
        if not self.args.model:
            raise ValueError("Model path required for export")
        
        self.model = SAM.load(self.args.model)
        
        # Export model
        from exporter import SAMExporter
        
        exporter = SAMExporter(self.model)
        
        export_format = self.args.export_format or "pytorch"
        output_path = self.args.output or f"sam_exported_{export_format}"
        
        if export_format == "pytorch":
            exporter.export_pytorch(output_path)
        elif export_format == "onnx":
            exporter.export_onnx(output_path)
        elif export_format == "tensorrt":
            exporter.export_tensorrt(output_path)
        elif export_format == "huggingface":
            exporter.export_huggingface(output_path)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        logger.info(f"Model exported to: {output_path}")
        return 0
    
    def _setup_distributed(self):
        """Setup distributed training"""
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA")
        
        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.args.world_size,
            rank=self.args.rank
        )
        
        # Set device
        torch.cuda.set_device(self.args.local_rank)
        
        logger.info(f"Initialized distributed training: rank {self.args.rank}/{self.args.world_size}")
    
    def _setup_data_loader(self, data_config: Dict):
        """Setup data loading"""
        if not self.args.data:
            raise ValueError("Data path required for training")
        
        data_path = Path(self.args.data)
        
        # Find processed data files
        if data_path.is_file():
            data_files = [data_path]
        elif data_path.is_dir():
            data_files = list(data_path.glob("*.jsonl"))
            if not data_files:
                raise ValueError(f"No .jsonl files found in {data_path}")
        else:
            raise ValueError(f"Data path not found: {data_path}")
        
        logger.info(f"Found {len(data_files)} data files")
        
        # Create dataset
        dataset = SAMDataset(
            data_files=data_files,
            max_length=data_config.get("max_sequence_length", 2048),
            model=self.model
        )
        
        # Create data loader
        data_loader = SAMDataLoader(
            dataset=dataset,
            batch_size=data_config.get("batch_size", 4),
            shuffle=data_config.get("shuffle", True),
            num_workers=data_config.get("num_workers", 4),
            distributed=self.args.distributed
        )
        
        return data_loader

def main():
    parser = argparse.ArgumentParser(description="SAM Training and Interaction")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True,
                       choices=["train", "interact", "evaluate", "export"],
                       help="Execution mode")
    
    # Common arguments
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data", type=str, help="Path to data directory or file")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    # Training arguments
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    parser.add_argument("--eval-steps", type=int, help="Evaluation frequency")
    parser.add_argument("--save-steps", type=int, help="Save frequency")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--no-evolution", action="store_true", help="Disable model evolution during training")
    
    # Distributed training arguments
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--world-size", type=int, default=1, help="Number of processes")
    parser.add_argument("--rank", type=int, default=0, help="Process rank")
    parser.add_argument("--local-rank", type=int, default=0, help="Local process rank")
    
    # Interactive arguments
    parser.add_argument("--interact-config", type=str, help="Interactive configuration file")
    parser.add_argument("--web-interface", action="store_true", help="Start web interface")
    parser.add_argument("--host", type=str, default="localhost", help="Web interface host")
    parser.add_argument("--port", type=int, default=8080, help="Web interface port")
    parser.add_argument("--no-save", action="store_true", help="Don't save conversations")
    
    # Evaluation arguments
    parser.add_argument("--eval-data", type=str, help="Evaluation data path")
    parser.add_argument("--eval-metrics", type=str, nargs="+", help="Evaluation metrics")
    
    # Export arguments
    parser.add_argument("--export-format", type=str, choices=["pytorch", "onnx", "tensorrt", "huggingface"],
                       help="Export format")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create runner and execute
    runner = SAMRunner(args)
    return runner.run()

if __name__ == "__main__":
    sys.exit(main())
