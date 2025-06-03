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

from arithmetic_model import ArithmeticModel, ArithmeticModelConfig
from arithmatic_dataset import (
    ArithmeticDatasetGenerator, 
    BinaryOp, 
    create_unified_dataloader,
    ArithmeticTokenizer
)

app = typer.Typer(help="Train ArithmeticModel using TransComputer")

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    max_digits: int = 2
    train_samples: int = 10000
    val_samples: int = 2000
    operations: List[str] = None  # Will default to ["ADD", "MUL", "SUB"]
    
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
    compile_model: bool = True
    
    # Logging
    log_every_steps: int = 100
    save_every_steps: int = 1000
    output_dir: str = "./checkpoints"
    
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
        
        # Create output directory
        os.makedirs(training_config.output_dir, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = ArithmeticTokenizer()
        
        # Setup data
        self._setup_data()
        
        # Setup model
        self._setup_model()
        
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
        
        # Update model config with correct number of operations
        self.model_config.num_operations = len(operations)
        
        # Create dataset generator
        self.generator = ArithmeticDatasetGenerator(
            max_digits=self.training_config.max_digits,
            operations=operations
        )
        
        # Generate datasets
        train_datasets, val_datasets = self.generator.create_datasets(
            train_samples=self.training_config.train_samples,
            val_samples=self.training_config.val_samples,
            auto_detect_lengths=True,
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
        """Generate example predictions for logging"""
        self.model.eval()
        examples = []
        
        # Operation names and symbols for display
        op_names = {0: "ADD", 1: "SUB", 2: "MUL"}
        op_symbols = {0: "+", 1: "-", 2: "*"}
        
        with torch.no_grad():
            # Get a few samples from validation set
            batch_dict = next(iter(self.val_loader))
            batch_dict = self._move_batch_to_device(batch_dict)
            
            count = 0
            for seq_len, batch_data in batch_dict.items():
                if count >= num_examples:
                    break
                
                token_ids = batch_data['token_ids']
                op_ids = batch_data['op_ids']
                input_lens = batch_data['input_lens']
                
                for i in range(min(token_ids.shape[0], num_examples - count)):
                    # Get input portion and parse the sequence
                    input_len = input_lens[i].item()
                    op_id = op_ids[i].item()
                    
                    # Parse operands for display
                    operands = self._extract_operands_from_sequence(token_ids[i], input_len)
                    
                    # Get operation info
                    op_name = op_names.get(op_id, f"OP{op_id}")
                    op_symbol = op_symbols.get(op_id, "?")
                    
                    # Get input tokens for generation
                    input_tokens = token_ids[i:i+1, :input_len]  # (1, input_len)
                    
                    # Generate solution
                    generated = self.model.generate(
                        input_tokens=input_tokens,
                        op_id=op_id,
                        input_len=input_len,
                        max_new_tokens=10,
                        temperature=0.1
                    )
                    
                    # Get target sequence (fixed length from dataset)
                    target_tokens = token_ids[i].cpu().tolist()
                    target_str = self.tokenizer.detokenize(target_tokens)
                    
                    # Get predicted sequence (ensure same length as target)
                    predicted_tokens = generated[0].cpu().tolist()
                    
                    # Pad or truncate predicted to match target length
                    if len(predicted_tokens) < len(target_tokens):
                        # Pad with padding tokens
                        predicted_tokens.extend([self.tokenizer.PAD_TOKEN] * (len(target_tokens) - len(predicted_tokens)))
                    elif len(predicted_tokens) > len(target_tokens):
                        # Truncate to target length
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
        checkpoint_path = os.path.join(self.training_config.output_dir, f"checkpoint_step_{self.step}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.training_config.output_dir, "best_model.pt")
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
            
            # Logging
            if step % self.training_config.log_every_steps == 0:
                elapsed = time.time() - start_time
                print(f"Step {step:5d} | Loss: {metrics['loss']:.4f} | "
                      f"SeqAcc: {metrics['sequence_accuracy']:.4f} | TokAcc: {metrics['token_accuracy']:.4f} | "
                      f"LR: {metrics['learning_rate']:.2e} | Time: {elapsed:.1f}s")
            
            # Validation
            if step % self.training_config.val_every_steps == 0 and step > 0:
                val_metrics = self.validate()
                self.validation_metrics.append({**val_metrics, 'step': step})
                
                print(f"Validation | Loss: {val_metrics['val_loss']:.4f} | "
                      f"SeqAcc: {val_metrics['val_sequence_accuracy']:.4f} | "
                      f"TokAcc: {val_metrics['val_token_accuracy']:.4f}")
                
                # Generate examples
                if self.training_config.generate_examples > 0:
                    examples = self.generate_examples(self.training_config.generate_examples)
                    print("Examples:")
                    for example in examples[:6]:  # Show first few
                        print(f"  {example}")
                
                # Save best model
                if val_metrics['val_accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['val_accuracy']
                    self.save_checkpoint(is_best=True)
            
            # Save checkpoint
            if step % self.training_config.save_every_steps == 0 and step > 0:
                self.save_checkpoint()
        
        print("Training completed!")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")


@app.command()
def main(
    max_digits: Annotated[int, typer.Option(help="Maximum digits for operands")] = 5,
    train_samples: Annotated[int, typer.Option(help="Number of training samples")] = 10000,
    val_samples: Annotated[int, typer.Option(help="Number of validation samples")] = 2000,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 1e-4,
    max_steps: Annotated[int, typer.Option(help="Maximum training steps")] = 10000,
    d_model: Annotated[int, typer.Option(help="Model dimension")] = 128,
    n_layers: Annotated[int, typer.Option(help="Number of layers")] = 6,
    n_heads: Annotated[int, typer.Option(help="Number of attention heads")] = 8,
    operations: Annotated[List[str], typer.Option(help="Operations to include (can be specified multiple times)")] = None,
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "./checkpoints",
    no_compile: Annotated[bool, typer.Option("--no-compile", help="Disable model compilation")] = False,
    # Training schedule parameters
    val_every_steps: Annotated[int, typer.Option(help="Run validation every N steps")] = 500,
    log_every_steps: Annotated[int, typer.Option(help="Log training metrics every N steps")] = 100,
    # TransComputer-specific parameters
    n_symbols: Annotated[int, typer.Option(help="Number of symbols in compute block symbol library")] = 100,
    include_ffn: Annotated[bool, typer.Option(help="Include feed-forward network in compute blocks")] = False,
    shared_compute_block: Annotated[bool, typer.Option(help="Share compute block parameters across layers")] = True,
    n_prog_tokens: Annotated[int, typer.Option(help="Number of program tokens")] = 16,
    compute_steps: Annotated[int, typer.Option(help="Number of compute steps per layer")] = 3,
):
    """Train ArithmeticModel using TransComputer."""
    
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
        output_dir=output_dir,
        compile_model=not no_compile,
        # Training schedule
        val_every_steps=val_every_steps,
        log_every_steps=log_every_steps,
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configs
    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(asdict(model_config), f, indent=2)
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(asdict(training_config), f, indent=2)
    
    # Create trainer and start training
    trainer = ArithmeticTrainer(model_config, training_config)
    trainer.train()


if __name__ == "__main__":
    app() 