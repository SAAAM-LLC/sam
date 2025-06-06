# trainer.py - SAM Training Components
import os
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging

logger = logging.getLogger(__name__)

class SAMTrainer:
    """Enhanced trainer for SAM models"""
    
    def __init__(self, model, config, data_loader, distributed=False, local_rank=0):
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.distributed = distributed
        self.local_rank = local_rank
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.shutdown_requested = False
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup model for training
        self.model.train()
        
        if distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank
            )
        
        # Mixed precision training
        self.use_amp = config.get('mixed_precision', True)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training configuration
        self.max_steps = config.get('max_steps', 100000)
        self.eval_steps = config.get('eval_steps', 1000)
        self.save_steps = config.get('save_steps', 5000)
        self.log_steps = config.get('log_steps', 100)
        self.gradient_clip = config.get('gradient_clip', 1.0)
        
        # Evolution settings
        self.evolve_every = config.get('evolve_every', 1000)
        self.enable_evolution = config.get('enable_evolution', True)
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        learning_rate = self.config.get('learning_rate', 3e-5)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        # Separate parameters for different learning rates
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if 'bias' in name or 'LayerNorm' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_params = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return AdamW(optimizer_params, lr=learning_rate)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        warmup_steps = self.config.get('warmup_steps', 1000)
        
        # Custom scheduler with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                # Cosine annealing after warmup
                progress = (step - warmup_steps) / (self.max_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.max_steps} steps")
        
        start_time = time.time()
        running_loss = 0.0
        
        while self.step < self.max_steps and not self.shutdown_requested:
            for batch in self.data_loader:
                if self.step >= self.max_steps or self.shutdown_requested:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                loss = self._forward_step(batch)
                
                # Backward pass
                self._backward_step(loss)
                
                running_loss += loss.item()
                self.step += 1
                
                # Logging
                if self.step % self.log_steps == 0:
                    avg_loss = running_loss / self.log_steps
                    lr = self.scheduler.get_last_lr()[0]
                    
                    logger.info(f"Step {self.step}/{self.max_steps} | "
                              f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                    
                    # Log to wandb if available
                    try:
                        import wandb
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/step': self.step
                        })
                    except:
                        pass
                    
                    running_loss = 0.0
                
                # Evolution
                if (self.enable_evolution and 
                    self.step % self.evolve_every == 0 and 
                    self.step > 0):
                    self._evolve_model()
                
                # Evaluation
                if self.step % self.eval_steps == 0:
                    self._evaluate()
                
                # Saving
                if self.step % self.save_steps == 0:
                    self._save_checkpoint()
        
        # Final save
        self._save_checkpoint(final=True)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
    
    def _forward_step(self, batch):
        """Forward step with loss computation"""
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch.get('attention_mask')
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_concepts=input_ids,
                    target_concepts=labels,
                    concept_mask=attention_mask,
                    return_dict=True
                )
                loss = outputs['loss']
        else:
            outputs = self.model(
                input_concepts=input_ids,
                target_concepts=labels,
                concept_mask=attention_mask,
                return_dict=True
            )
            loss = outputs['loss']
        
        return loss
    
    def _backward_step(self, loss):
        """Backward step with gradient updates"""
        if self.use_amp:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
    
    def _evolve_model(self):
        """Evolve the model"""
        logger.info(f"Evolving model at step {self.step}")
        
        try:
            # Get the actual model (unwrap DDP if needed)
            model = self.model.module if self.distributed else self.model
            
            # Perform evolution
            evolution_results = model.evolve()
            
            logger.info(f"Evolution completed: {evolution_results}")
            
            # Log evolution results
            try:
                import wandb
                wandb.log({
                    'evolution/step': self.step,
                    'evolution/results': evolution_results
                })
            except:
                pass
                
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
    
    def _evaluate(self):
        """Evaluate the model"""
        # Placeholder for evaluation logic
        logger.info(f"Evaluation at step {self.step} (placeholder)")
    
    def _save_checkpoint(self, final=False):
        """Save model checkpoint"""
        if self.distributed and self.local_rank != 0:
            return  # Only save on rank 0
        
        # Get the actual model (unwrap DDP if needed)
        model = self.model.module if self.distributed else self.model
        
        if final:
            checkpoint_path = f"models/checkpoints/sam_final"
        else:
            checkpoint_path = f"models/checkpoints/sam_step_{self.step}"
        
        try:
            model.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Keep track of best model
            # (This would require implementing proper evaluation)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def request_shutdown(self):
        """Request graceful shutdown"""
        self.shutdown_requested = True
