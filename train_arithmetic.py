"""
Training script for ArithmeticModel using TransComputer.

Features:
- Automatic device detection (CUDA/CPU)
- Model compilation for performance
- Validation every N steps
- Proper loss computation on output tokens only
- Configurable number of operations
- Adam optimizer
- Comprehensive logging
- Weights & Biases integration
- Flexible project/run organization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import typer
from typing_extensions import Annotated
from rich import print

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from arithmetic_model import ArithmeticModel, ArithmeticModelConfig
from arithmatic_dataset import (
    ArithmeticDatasetGenerator, 
    BinaryOp, 
    create_unified_dataloader,
    ArithmeticTokenizer
)

app = typer.Typer(help="Train ArithmeticModel using TransComputer", pretty_exceptions_show_locals=False)

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Experiment organization
    project: str = "transcomputer-arithmetic"
    run_name: str = "baseline"
    
    # Data
    max_digits: int = 2
    train_samples: int = 10000
    val_samples: int = 2000
    operations: List[str] = None  # Will default to ["ADD", "MUL", "SUB"]
    
    # Data generation optimization
    use_cache: bool = True
    num_workers: int = None  # None for auto-detection
    max_workers: int = 4  # Conservative default for memory-limited systems
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 1000
    
    # Validation
    val_every_steps: int = 500
    val_steps: int = 100
    
    # Model compilation
    compile_model: bool = False
    
    # Logging
    log_every_steps: int = 100
    save_every_steps: int = 1000
    use_wandb: bool = True
    
    # Generation (for validation)
    generate_examples: int = 5
    
    def __post_init__(self):
        if self.operations is None:
            self.operations = ["ADD", "MUL", "SUB"]


class ArithmeticTrainer:
    """Trainer for ArithmeticModel"""
    
    def __init__(self, 
                 model_config: ArithmeticModelConfig,
                 training_config: TrainingConfig,
                 device: Optional[str] = None):
        self.model_config = model_config
        self.training_config = training_config
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA version: {torch.version.cuda}")
        
        # Create output directory using project/run_name structure
        self.output_dir = os.path.join("experiments", training_config.project, training_config.run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Experiment directory: {self.output_dir}")
        
        # Initialize wandb if available and requested
        self.use_wandb = training_config.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            # Create wandb config combining both model and training configs
            wandb_config = {
                **asdict(model_config),
                **asdict(training_config),
                "device": str(self.device),
            }
            
            wandb.init(
                project=training_config.project,
                name=training_config.run_name,
                config=wandb_config,
                save_code=True
            )
            print(f"Initialized wandb: {training_config.project}/{training_config.run_name}")
        elif training_config.use_wandb:
            print("Warning: wandb requested but not available. Continuing without wandb logging.")
        
        # Initialize tokenizer
        self.tokenizer = ArithmeticTokenizer()
        
        # Setup data
        self._setup_data()
        
        # Setup model
        self._setup_model()
        
        # Update wandb config with model info
        if self.use_wandb:
            wandb.config.update({
                "total_params": self.model.get_num_params(),
                "trainable_params": self.model.get_num_trainable_params(),
            }, allow_val_change=True)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Training state
        self.step = 0
        self.best_val_accuracy = 0.0
        self.training_metrics = []
        self.validation_metrics = []
    
    def _setup_data(self):
        """Setup training and validation data"""
        print("Setting up data...")
        
        # Convert operation strings to BinaryOp enums
        op_mapping = {"ADD": BinaryOp.ADD, "SUB": BinaryOp.SUB, "MUL": BinaryOp.MUL}
        operations = [op_mapping[op] for op in self.training_config.operations]
        
        # Create dataset generator
        self.generator = ArithmeticDatasetGenerator(
            max_digits=self.training_config.max_digits,
            operations=operations
        )
        
        # Update model config with correct number of operations and theoretical max sequence length
        self.model_config.num_operations = len(operations)
        self.model_config.max_seq_len = self.generator.max_seq_len
        print(f"Set max_seq_len to {self.model_config.max_seq_len} (theoretical max based on {self.training_config.max_digits} digits)")
        
        # Generate datasets using theoretical max sequence length
        train_datasets, val_datasets = self.generator.create_datasets(
            train_samples=self.training_config.train_samples,
            val_samples=self.training_config.val_samples,
            auto_detect_lengths=False,  # Use theoretical max instead
            use_cache=self.training_config.use_cache,
            num_workers=self.training_config.num_workers,
            seed=42
        )
        
        # Create unified DataLoaders
        self.train_loader = create_unified_dataloader(
            train_datasets, 
            batch_size=self.training_config.batch_size, 
            shuffle=True
        )
        
        self.val_loader = create_unified_dataloader(
            val_datasets,
            batch_size=self.training_config.batch_size,
            shuffle=False
        )
        
        print(f"Training samples: {sum(len(ds) for ds in train_datasets.values())}")
        print(f"Validation samples: {sum(len(ds) for ds in val_datasets.values())}")
        print(f"Sequence length buckets: {list(train_datasets.keys())}")
    
    def _setup_model(self):
        """Setup model and move to device"""
        print("Setting up model...")
        
        # Create model
        self.model = ArithmeticModel(self.model_config).to(self.device)
        
        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"Trainable parameters: {self.model.get_num_trainable_params():,}")
        
        # Compile model for performance (if requested and supported)
        if self.training_config.compile_model:
            # DEBUGGING: Uncomment the next line to temporarily disable compilation
            # self.training_config.compile_model = False
            try:
                print("Compiling model with inductor backend...")
                # Use inductor backend with maximum optimizations
                self.model.forward = torch.compile(
                    self.model.forward,
                    backend="inductor",
                    mode="reduce-overhead",
                    fullgraph=True
                )
                print("Model forward compiled successfully with inductor backend!")
            except Exception as e:
                print(f"Model compilation failed: {e}")
                print("Continuing without compilation...")
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        print("Setting up optimizer...")
        
        # Adam optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < self.training_config.warmup_steps:
                return step / self.training_config.warmup_steps
            else:
                # Cosine decay after warmup
                progress = (step - self.training_config.warmup_steps) / (
                    self.training_config.max_steps - self.training_config.warmup_steps
                )
                return 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi))).item()
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _move_batch_to_device(self, batch_dict: Dict) -> Dict:
        """Move batch data to device"""
        device_batch = {}
        for seq_len, batch_data in batch_dict.items():
            device_batch[seq_len] = {
                'token_ids': batch_data['token_ids'].to(self.device),
                'op_ids': batch_data['op_ids'].to(self.device),
                'input_lens': batch_data['input_lens'].to(self.device),
                'batch_size': batch_data['batch_size']
            }
        return device_batch
    
    def train_step(self, batch_dict: Dict) -> Dict[str, float]:
        """Single training step using optimized forward pass"""
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        total_sequence_accuracy = 0.0
        total_token_accuracy = 0.0
        total_sequences = 0
        total_tokens = 0
        num_groups = 0
        
        # Move batch to device
        batch_dict = self._move_batch_to_device(batch_dict)
        
        # Process each sequence length group
        for seq_len, batch_data in batch_dict.items():
            # Use optimized forward pass with loss computation
            loss, logits, mask = self.model.forward(
                token_ids=batch_data['token_ids'],
                op_ids=batch_data['op_ids'],
                input_lens=batch_data['input_lens'],
                return_loss=True
            )
            
            # Compute metrics outside of compiled graph
            targets = batch_data['token_ids'][:, 1:]
            metrics = self.model.compute_metrics(loss, logits, targets, batch_data['input_lens'], mask)
            
            batch_size = batch_data['batch_size']
            total_loss += loss
            total_sequence_accuracy += metrics['sequence_accuracy'] * batch_size
            total_token_accuracy += metrics['token_accuracy'] * metrics['num_output_tokens']
            total_sequences += batch_size
            total_tokens += metrics['num_output_tokens']
            num_groups += 1
        
        # Backward pass
        if total_tokens > 0:
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            avg_sequence_accuracy = total_sequence_accuracy / total_sequences if total_sequences > 0 else 0.0
            avg_token_accuracy = total_token_accuracy / total_tokens if total_tokens > 0 else 0.0
        else:
            avg_sequence_accuracy = 0.0
            avg_token_accuracy = 0.0
        
        return {
            'loss': total_loss.item(),
            'accuracy': avg_sequence_accuracy,  # Main accuracy is sequence-level
            'sequence_accuracy': avg_sequence_accuracy,
            'token_accuracy': avg_token_accuracy,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'num_tokens': total_tokens,
            'num_sequences': total_sequences
        }
    
    def validate(self) -> Dict[str, float]:
        """Run validation using optimized forward pass"""
        self.model.eval()
        
        total_loss = 0.0
        total_sequence_accuracy = 0.0
        total_token_accuracy = 0.0
        total_sequences = 0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(self.val_loader):
                if batch_idx >= self.training_config.val_steps:
                    break
                
                # Move batch to device
                batch_dict = self._move_batch_to_device(batch_dict)
                
                # Process each sequence length group
                for seq_len, batch_data in batch_dict.items():
                    # Use optimized forward pass
                    loss, logits, mask = self.model.forward(
                        token_ids=batch_data['token_ids'],
                        op_ids=batch_data['op_ids'],
                        input_lens=batch_data['input_lens'],
                        return_loss=True
                    )
                    
                    # Compute metrics outside of compiled graph
                    targets = batch_data['token_ids'][:, 1:]
                    metrics = self.model.compute_metrics(loss, logits, targets, batch_data['input_lens'], mask)
                    
                    batch_size = batch_data['batch_size']
                    total_loss += loss.item()
                    total_sequence_accuracy += metrics['sequence_accuracy'] * batch_size
                    total_token_accuracy += metrics['token_accuracy'] * metrics['num_output_tokens']
                    total_sequences += batch_size
                    total_tokens += metrics['num_output_tokens']
                
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_sequence_accuracy = total_sequence_accuracy / total_sequences if total_sequences > 0 else 0.0
        avg_token_accuracy = total_token_accuracy / total_tokens if total_tokens > 0 else 0.0
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': avg_sequence_accuracy,
            'val_sequence_accuracy': avg_sequence_accuracy,
            'val_token_accuracy': avg_token_accuracy,
            'val_tokens': total_tokens,
            'val_sequences': total_sequences
        }
    
    def generate_examples(self, num_examples: int = 5) -> List[str]:
        """Generate example predictions using teacher-forced evaluation (not autoregressive)"""
        self.model.eval()
        examples = []
        
        # Operation names and symbols for display
        op_names = {0: "ADD", 1: "SUB", 2: "MUL"}
        op_symbols = {0: "+", 1: "-", 2: "*"}
        
        with torch.no_grad():
            # Check if validation loader has any batches
            try:
                batch_dict = next(iter(self.val_loader))
                batch_dict = self._move_batch_to_device(batch_dict)
            except StopIteration:
                # No validation batches available (due to drop_last=True)
                # Use training data instead
                batch_dict = next(iter(self.train_loader))
                batch_dict = self._move_batch_to_device(batch_dict)
            
            count = 0
            for seq_len, batch_data in batch_dict.items():
                if count >= num_examples:
                    break
                
                token_ids = batch_data['token_ids']
                op_ids = batch_data['op_ids']
                input_lens = batch_data['input_lens']
                
                # Use teacher-forced evaluation (same as validation)
                loss, logits, mask = self.model.forward(
                    token_ids=token_ids,
                    op_ids=op_ids,
                    input_lens=input_lens,
                    return_loss=True
                )
                
                # Get predictions from logits
                predictions = torch.argmax(logits, dim=-1)  # (B, T-1)
                
                for i in range(min(token_ids.shape[0], num_examples - count)):
                    # Get input portion and parse the sequence
                    input_len = input_lens[i].item()
                    op_id = op_ids[i].item()
                    
                    # Parse operands for display
                    operands = self._extract_operands_from_sequence(token_ids[i], input_len)
                    
                    # Get operation info
                    op_name = op_names.get(op_id, f"OP{op_id}")
                    op_symbol = op_symbols.get(op_id, "?")
                    
                    # Get target sequence (ground truth)
                    target_tokens = token_ids[i].cpu().tolist()
                    target_str = self.tokenizer.detokenize(target_tokens)
                    
                    # Get predicted sequence (from teacher-forced logits)
                    # Reconstruct full sequence: input + predicted output
                    input_tokens = token_ids[i, :input_len].cpu().tolist()
                    predicted_output = predictions[i].cpu().tolist()  # This is shifted by 1
                    
                    # The predictions correspond to token_ids[:, 1:], so we need to construct
                    # the full predicted sequence properly
                    predicted_tokens = input_tokens + predicted_output[input_len-1:]
                    
                    # Ensure same length as target
                    if len(predicted_tokens) < len(target_tokens):
                        predicted_tokens.extend([self.tokenizer.PAD_TOKEN] * (len(target_tokens) - len(predicted_tokens)))
                    elif len(predicted_tokens) > len(target_tokens):
                        predicted_tokens = predicted_tokens[:len(target_tokens)]
                    
                    predicted_str = self.tokenizer.detokenize(predicted_tokens)
                    
                    # Format the operation description
                    if operands and len(operands) >= 2:
                        operation_desc = f"{operands[0]} {op_symbol} {operands[1]} [{op_name}]"
                    else:
                        operation_desc = f"[{op_name}]"
                    
                    # Simple one-line format
                    examples.append(f"{operation_desc}: Target='{target_str}', Predicted='{predicted_str}'")
                    
                    count += 1
                    if count >= num_examples:
                        break
        
        return examples
    
    def _extract_operands_from_sequence(self, token_ids: torch.Tensor, input_len: int) -> List[str]:
        """Extract operands from input portion of sequence"""
        try:
            # Get input tokens (before "=")
            input_tokens = token_ids[:input_len].cpu().tolist()
            input_str = self.tokenizer.detokenize(input_tokens)
            
            # Split by spaces and filter out empty strings
            parts = [part for part in input_str.split(' ') if part.strip()]
            
            # Should have exactly 2 operands
            if len(parts) >= 2:
                return [parts[0], parts[1]]
            return []
        except:
            return []
    
    def _extract_result_from_sequence(self, token_ids: torch.Tensor, input_len: int) -> str:
        """Extract expected result from full sequence"""
        try:
            # Get tokens after "= " (input_len points to "=", so skip "= ")
            result_start = input_len + 2
            result_tokens = token_ids[result_start:].cpu().tolist()
            
            # Remove padding tokens
            result_tokens = [t for t in result_tokens if t != self.tokenizer.PAD_TOKEN]
            
            if result_tokens:
                return self.tokenizer.detokenize(result_tokens).strip()
            return ""
        except:
            return ""
    
    def _extract_generated_result(self, generated_tokens: torch.Tensor, input_len: int) -> str:
        """Extract result from generated sequence"""
        try:
            # Find the "= " and extract what comes after
            generated_list = generated_tokens.cpu().tolist()
            
            # Look for equals token
            equals_pos = None
            for i, token in enumerate(generated_list):
                if token == self.tokenizer.EQUALS_TOKEN:
                    equals_pos = i
                    break
            
            if equals_pos is not None:
                # Extract tokens after "= "
                result_start = equals_pos + 2  # Skip "= "
                result_tokens = generated_list[result_start:]
                
                # Remove padding tokens
                result_tokens = [t for t in result_tokens if t != self.tokenizer.PAD_TOKEN]
                
                if result_tokens:
                    return self.tokenizer.detokenize(result_tokens).strip()
            
            return ""
        except:
            return ""
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': asdict(self.model_config),
            'training_config': asdict(self.training_config),
            'best_val_accuracy': self.best_val_accuracy,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step_{self.step}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved with accuracy: {self.best_val_accuracy:.4f}")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Max steps: {self.training_config.max_steps}")
        print(f"Validation every {self.training_config.val_every_steps} steps")
        print(f"Logging every {self.training_config.log_every_steps} steps")
        
        start_time = time.time()
        
        # Create infinite iterator over training data
        train_iter = iter(self.train_loader)
        
        for step in range(self.training_config.max_steps):
            self.step = step
            
            try:
                batch_dict = next(train_iter)
            except StopIteration:
                # Restart iterator
                train_iter = iter(self.train_loader)
                batch_dict = next(train_iter)
            
            # Training step
            metrics = self.train_step(batch_dict)
            self.training_metrics.append({**metrics, 'step': step})
            
            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    "train/loss": metrics['loss'],
                    "train/sequence_accuracy": metrics['sequence_accuracy'],
                    "train/token_accuracy": metrics['token_accuracy'],
                    "train/learning_rate": metrics['learning_rate'],
                    "train/num_tokens": metrics['num_tokens'],
                    "train/num_sequences": metrics['num_sequences'],
                    "step": step
                }
                
                # Add gate alphas if in TransComputer mode and we have program tokens
                if hasattr(self.model, 'transcomputer') and self.model_config.n_prog_tokens > 0:
                    gate_alphas = self.get_gate_alphas()
                    for key, value in gate_alphas.items():
                        log_dict[f"gates/{key}"] = value
                
                wandb.log(log_dict)
            
            # Console logging
            if step % self.training_config.log_every_steps == 0:
                elapsed = time.time() - start_time
                print(f"Step {step:5d} | Loss: {metrics['loss']:.4f} | "
                      f"SeqAcc: {metrics['sequence_accuracy']:.4f} | TokAcc: {metrics['token_accuracy']:.4f} | "
                      f"LR: {metrics['learning_rate']:.2e} | Time: {elapsed:.1f}s")
            
            # Validation
            if step % self.training_config.val_every_steps == 0 and step > 0:
                val_metrics = self.validate()
                self.validation_metrics.append({**val_metrics, 'step': step})
                
                # Log validation metrics to wandb
                if self.use_wandb:
                    val_log_dict = {
                        "val/loss": val_metrics['val_loss'],
                        "val/sequence_accuracy": val_metrics['val_sequence_accuracy'],
                        "val/token_accuracy": val_metrics['val_token_accuracy'],
                        "val/num_tokens": val_metrics['val_tokens'],
                        "val/num_sequences": val_metrics['val_sequences'],
                        "step": step
                    }
                    
                    # Add gate alphas during validation as well if we have program tokens
                    if hasattr(self.model, 'transcomputer') and self.model_config.n_prog_tokens > 0:
                        gate_alphas = self.get_gate_alphas()
                        for key, value in gate_alphas.items():
                            val_log_dict[f"gates_val/{key}"] = value
                    
                    wandb.log(val_log_dict)
                
                print(f"Validation | Loss: {val_metrics['val_loss']:.4f} | "
                      f"SeqAcc: {val_metrics['val_sequence_accuracy']:.4f} | "
                      f"TokAcc: {val_metrics['val_token_accuracy']:.4f}")
                
                # Log gate alphas to console if in TransComputer mode and we have program tokens
                if hasattr(self.model, 'transcomputer') and self.model_config.n_prog_tokens > 0:
                    gate_alphas = self.get_gate_alphas()
                    print("Gate Alphas:")
                    print(f"  Perception→Compute: XAttn={gate_alphas['perception_to_compute/avg_cross_attn_gate']:.4f}, "
                          f"FFN={gate_alphas['perception_to_compute/avg_ffn_gate']:.4f}")
                    print(f"  Compute→Perception: XAttn={gate_alphas['compute_to_perception/avg_cross_attn_gate']:.4f}, "
                          f"FFN={gate_alphas['compute_to_perception/avg_ffn_gate']:.4f}")
                
                # Generate examples
                if self.training_config.generate_examples > 0:
                    examples = self.generate_examples(self.training_config.generate_examples)
                    print("Examples:")
                    for example in examples[:6]:  # Show first few
                        print(f"  {example}")
                    
                    # Log examples to wandb as a table
                    if self.use_wandb and examples:
                        example_table = wandb.Table(columns=["Example"])
                        for example in examples:
                            example_table.add_data(example)
                        wandb.log({"examples": example_table, "step": step})
                
                # Save best model
                if val_metrics['val_accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['val_accuracy']
                    self.save_checkpoint(is_best=True)
                    
                    # Log best accuracy to wandb
                    if self.use_wandb:
                        wandb.log({"best_val_accuracy": self.best_val_accuracy, "step": step})
            
            # Save checkpoint
            if step % self.training_config.save_every_steps == 0 and step > 0:
                self.save_checkpoint()
        
        print("Training completed!")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        # Finish wandb run
        if self.use_wandb:
            wandb.finish()

    def get_gate_alphas(self) -> Dict[str, float]:
        """Extract gate alpha values from gated cross attention blocks"""
        gate_alphas = {}
        
        # Extract gates if the model has TransComputer (gates exist regardless of n_prog_tokens)
        if hasattr(self.model, 'transcomputer'):
            transcomputer = self.model.transcomputer
            
            # Extract perception->compute gate alphas (perception_gated_xattn)
            for i in range(len(transcomputer.perception_gated_xattn)):
                gated_xattn = transcomputer.perception_gated_xattn[i]
                # Gate values after tanh activation
                gate_xattn_val = torch.tanh(gated_xattn.gate_xattn).item()
                gate_dense_val = torch.tanh(gated_xattn.gate_dense).item()
                
                gate_alphas[f"perception_to_compute/layer_{i}/cross_attn_gate"] = gate_xattn_val
                gate_alphas[f"perception_to_compute/layer_{i}/ffn_gate"] = gate_dense_val
            
            # Extract compute->perception gate alphas (compute_gated_xattn)
            for i in range(len(transcomputer.compute_gated_xattn)):
                gated_xattn = transcomputer.compute_gated_xattn[i]
                # Gate values after tanh activation
                gate_xattn_val = torch.tanh(gated_xattn.gate_xattn).item()
                gate_dense_val = torch.tanh(gated_xattn.gate_dense).item()
                
                gate_alphas[f"compute_to_perception/layer_{i}/cross_attn_gate"] = gate_xattn_val
                gate_alphas[f"compute_to_perception/layer_{i}/ffn_gate"] = gate_dense_val
            
            # Compute average gate values across layers
            if len(gate_alphas) > 0:  # Only compute averages if we have gates
                perception_to_compute_xattn_avg = sum(gate_alphas[k] for k in gate_alphas.keys() 
                                                     if "perception_to_compute" in k and "cross_attn_gate" in k) / self.model_config.n_layers
                perception_to_compute_ffn_avg = sum(gate_alphas[k] for k in gate_alphas.keys() 
                                                   if "perception_to_compute" in k and "ffn_gate" in k) / self.model_config.n_layers
                compute_to_perception_xattn_avg = sum(gate_alphas[k] for k in gate_alphas.keys() 
                                                     if "compute_to_perception" in k and "cross_attn_gate" in k) / self.model_config.n_layers
                compute_to_perception_ffn_avg = sum(gate_alphas[k] for k in gate_alphas.keys() 
                                                   if "compute_to_perception" in k and "ffn_gate" in k) / self.model_config.n_layers
                
                gate_alphas["perception_to_compute/avg_cross_attn_gate"] = perception_to_compute_xattn_avg
                gate_alphas["perception_to_compute/avg_ffn_gate"] = perception_to_compute_ffn_avg
                gate_alphas["compute_to_perception/avg_cross_attn_gate"] = compute_to_perception_xattn_avg
                gate_alphas["compute_to_perception/avg_ffn_gate"] = compute_to_perception_ffn_avg
        
        return gate_alphas


@app.command()
def main(
    # Experiment organization
    project: Annotated[str, typer.Option(help="Project name for organizing experiments")] = "transcomputer-arithmetic",
    run_name: Annotated[str, typer.Option(help="Run name for this specific experiment")] = "baseline",
    use_wandb: Annotated[bool, typer.Option("--wandb/--no-wandb", help="Enable/disable Weights & Biases logging")] = True,
    
    # Data parameters
    max_digits: Annotated[int, typer.Option(help="Maximum digits for operands")] = 2,
    train_samples: Annotated[int, typer.Option(help="Number of training samples")] = 10000,
    val_samples: Annotated[int, typer.Option(help="Number of validation samples")] = 2000,
    operations: Annotated[List[str], typer.Option(help="Operations to include (can be specified multiple times)")] = None,
    
    # Data generation optimization
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Disable dataset caching")] = False,
    num_workers: Annotated[Optional[int], typer.Option(help="Number of parallel workers for dataset generation (None for auto)")] = None,
    max_workers: Annotated[int, typer.Option(help="Maximum number of workers (useful for limited memory systems)")] = 4,
    
    # Training parameters
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 1e-4,
    max_steps: Annotated[int, typer.Option(help="Maximum training steps")] = 10000,
    val_every_steps: Annotated[int, typer.Option(help="Run validation every N steps")] = 500,
    log_every_steps: Annotated[int, typer.Option(help="Log training metrics every N steps")] = 100,
    no_compile: Annotated[bool, typer.Option("--no-compile", help="Disable model compilation")] = False,
    
    # Model architecture parameters
    d_model: Annotated[int, typer.Option(help="Model dimension")] = 128,
    n_layers: Annotated[int, typer.Option(help="Number of layers")] = 6,
    n_heads: Annotated[int, typer.Option(help="Number of attention heads")] = 8,
    
    # TransComputer-specific parameters (key experimental variables)
    n_prog_tokens: Annotated[int, typer.Option(help="Number of program tokens (0=vanilla transformer)")] = 0,
    compute_steps: Annotated[int, typer.Option(help="Number of compute steps per layer")] = 1,
    n_symbols: Annotated[int, typer.Option(help="Number of symbols in compute block symbol library")] = 100,
    include_ffn: Annotated[bool, typer.Option(help="Include feed-forward network in compute blocks")] = False,
    shared_compute_block: Annotated[bool, typer.Option(help="Share compute block parameters across layers")] = True,
):
    """
    Train ArithmeticModel using TransComputer.
    
    This script is designed for studying the effect of computational blocks.
    
    Key experimental parameters:
    - n_prog_tokens: 0 = vanilla transformer, >0 = TransComputer
    - compute_steps: Number of computation iterations per layer
    
    Examples:
    
    Baseline (vanilla transformer):
    python train_arithmetic.py --project "compute-study" --run-name "baseline" --n-prog-tokens 0
    
    TransComputer with varying program tokens:
    python train_arithmetic.py --project "compute-study" --run-name "prog-tokens-8" --n-prog-tokens 8
    python train_arithmetic.py --project "compute-study" --run-name "prog-tokens-16" --n-prog-tokens 16
    
    TransComputer with varying compute steps:
    python train_arithmetic.py --project "compute-study" --run-name "compute-steps-3" --n-prog-tokens 8 --compute-steps 3
    """
    
    # Set default operations if none provided
    if operations is None:
        operations = ["ADD", "MUL", "SUB"]
    
    # Validate operations
    valid_operations = {"ADD", "SUB", "MUL"}
    for op in operations:
        if op not in valid_operations:
            typer.echo(f"Error: Invalid operation '{op}'. Valid operations are: {', '.join(valid_operations)}")
            raise typer.Exit(1)
    
    # Create configs
    model_config = ArithmeticModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        num_operations=len(operations),
        # TransComputer-specific parameters
        n_symbols=n_symbols,
        include_ffn=include_ffn,
        shared_compute_block=shared_compute_block,
        n_prog_tokens=n_prog_tokens,
        compute_steps=compute_steps,
    )
    
    training_config = TrainingConfig(
        max_digits=max_digits,
        train_samples=train_samples,
        val_samples=val_samples,
        operations=operations,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        project=project,
        run_name=run_name,
        compile_model=not no_compile,
        # Training schedule
        val_every_steps=val_every_steps,
        log_every_steps=log_every_steps,
        use_wandb=use_wandb,
        # Data generation optimization
        use_cache=not no_cache,
        num_workers=num_workers,
        max_workers=max_workers,
    )
    
    # Create trainer (this will create the output directory)
    trainer = ArithmeticTrainer(model_config, training_config)
    
    # Save configs to the trainer's output directory
    with open(os.path.join(trainer.output_dir, "model_config.json"), "w") as f:
        json.dump(asdict(model_config), f, indent=2)
    with open(os.path.join(trainer.output_dir, "training_config.json"), "w") as f:
        json.dump(asdict(training_config), f, indent=2)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    app() 