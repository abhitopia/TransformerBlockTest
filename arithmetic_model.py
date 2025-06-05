"""
ArithmeticModel: A wrapper around TransComputer for arithmetic reasoning tasks.
Includes token embeddings, language modeling head, and proper loss computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

from TransComputer import TransComputer, Config as TransComputerConfig
from arithmatic_dataset import ArithmeticTokenizer, BinaryOp


@dataclass
class ArithmeticModelConfig:
    """Configuration for ArithmeticModel"""
    # Model architecture
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 32  # Sufficient for arithmetic tasks (longest sequences ~16 tokens)
    
    # TransComputer specific
    n_symbols: int = 100
    include_ffn: bool = False
    shared_compute_block: bool = True
    n_prog_tokens: int = 16
    compute_steps: int = 3
    causal_compute: bool = False  # Enable causal attention in compute blocks
    
    # Task specific
    vocab_size: int = 13  # From ArithmeticTokenizer
    num_operations: int = 3  # ADD, SUB, MUL (configurable)
    
    # Training
    causal: bool = True


class ArithmeticModel(nn.Module):
    """
    Complete model for arithmetic reasoning using TransComputer.
    
    Includes:
    - Token embeddings
    - TransComputer backbone
    - Language modeling head
    - Proper loss computation for arithmetic tasks
    """
    
    def __init__(self, config: ArithmeticModelConfig):
        super().__init__()
        self.config = config
        self.tokenizer = ArithmeticTokenizer()
        
        # Verify vocab size matches tokenizer
        assert config.vocab_size == self.tokenizer.vocab_size, \
            f"Config vocab_size ({config.vocab_size}) != tokenizer vocab_size ({self.tokenizer.vocab_size})"
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional embeddings
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # TransComputer configuration
        tc_config = TransComputerConfig(
            n_layers=config.n_layers,
            causal=config.causal,
            n_symbols=config.n_symbols,
            include_ffn=config.include_ffn,
            shared_compute_block=config.shared_compute_block,
            n_prog_tokens=config.n_prog_tokens,
            n_programs=config.num_operations,
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            causal_compute=config.causal_compute
        )
        
        # TransComputer backbone
        self.transcomputer = TransComputer(tc_config)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights following best practices"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, 
                token_ids: torch.Tensor,
                op_ids: torch.Tensor,
                input_lens: torch.Tensor,
                compute_steps: Optional[int] = None,
                return_loss: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional loss computation.
        
        Args:
            token_ids: (B, T) token sequences
            op_ids: (B,) operation IDs
            input_lens: (B,) lengths until "=" (exclusive)
            compute_steps: Number of compute iterations (optional)
            return_loss: Whether to compute and return loss
            
        Returns:
            If return_loss=False: logits (B, T, vocab_size)
            If return_loss=True: (loss, logits, mask) where mask indicates output tokens
        """
        # Embed tokens
        x = self.token_embedding(token_ids)  # (B, T, d_model)
        
        # Positional embeddings (create position IDs dynamically based on input size)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)  # (B, T)
        pos_embedding = self.pos_embedding(positions)  # (B, T, d_model)
        
        # Forward through TransComputer
        hidden_states = self.transcomputer.forward(
            x=x + pos_embedding,
            prog_ids=op_ids,
            input_lens=input_lens,
            compute_steps=compute_steps or self.config.compute_steps
        )  # (B, T, d_model)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)  # (B, T, vocab_size)
        
        if not return_loss:
            return logits
        
        # Compute loss (we always have output tokens in arithmetic dataset)
        batch_size, seq_len = token_ids.shape
        
        # Standard language modeling: predict next token
        targets = token_ids[:, 1:]  # (B, T-1)
        logits_shifted = logits[:, :-1, :]  # (B, T-1, vocab_size)
        
        # Create mask for output tokens (after "=" + space)
        start_positions = input_lens  # (B,) - Start from space after "=" (fixed: was input_lens + 1)
        positions = torch.arange(targets.size(1), device=targets.device).unsqueeze(0)  # (1, T-1)
        mask = positions >= start_positions.unsqueeze(1)  # (B, T-1)
        
        # Flatten everything first
        flat_logits = logits_shifted.reshape(-1, logits_shifted.size(-1))  # (B*(T-1), vocab_size)
        flat_targets = targets.reshape(-1)  # (B*(T-1),)
        flat_mask = mask.reshape(-1).float()  # (B*(T-1),) - convert to float for weighting
        
        # Compute loss using mask as weights (compilation-friendly)
        # This avoids boolean indexing which creates dynamic shapes
        unreduced_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')  # (B*(T-1),)
        masked_loss = unreduced_loss * flat_mask  # Apply mask as weights
        
        # Compute mean over non-masked positions
        loss = masked_loss.sum() / flat_mask.sum().clamp(min=1.0)  # Avoid division by zero
        
        return loss, logits_shifted, mask
    
    def compute_metrics(self,
                       loss: torch.Tensor,
                       logits: torch.Tensor,
                       targets: torch.Tensor,
                       input_lens: torch.Tensor,
                       mask: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics outside of compiled graph to avoid graph breaks.
        
        Args:
            loss: Loss tensor
            logits: (B, T-1, vocab_size) model predictions
            targets: (B, T-1) shifted target tokens
            input_lens: (B,) input lengths
            mask: (B, T-1) mask for output tokens
            
        Returns:
            Dictionary of metrics
        """
        if mask.any():
            # Flatten and apply mask for token accuracy
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            flat_mask = mask.reshape(-1)
            
            masked_logits = flat_logits[flat_mask]
            masked_targets = flat_targets[flat_mask]
            
            # Token-level accuracy
            predictions = masked_logits.argmax(dim=-1)
            token_accuracy = (predictions == masked_targets).float().mean().item()
            
            # Sequence-level accuracy
            sequence_accuracy = self._compute_sequence_accuracy_simple(
                logits, targets, input_lens, mask
            )
            
            return {
                'loss': loss.item(),
                'accuracy': sequence_accuracy,
                'token_accuracy': token_accuracy,
                'sequence_accuracy': sequence_accuracy,
                'num_output_tokens': len(masked_targets)
            }
        else:
            return {
                'loss': loss.item(),
                'accuracy': 0.0,
                'token_accuracy': 0.0,
                'sequence_accuracy': 0.0,
                'num_output_tokens': 0
            }

    def compute_loss(self,
                    token_ids: torch.Tensor,
                    op_ids: torch.Tensor,
                    input_lens: torch.Tensor,
                    compute_steps: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Legacy compute_loss method for backward compatibility.
        Now calls the new forward method and computes metrics separately.
        """
        # Get loss and intermediate results from forward
        loss, logits, mask = self.forward(
            token_ids=token_ids,
            op_ids=op_ids,
            input_lens=input_lens,
            compute_steps=compute_steps,
            return_loss=True
        )
        
        # Compute metrics outside of compiled graph
        targets = token_ids[:, 1:]
        metrics = self.compute_metrics(loss, logits, targets, input_lens, mask)
        
        return loss, metrics
    
    def _compute_sequence_accuracy_simple(self, 
                                        logits: torch.Tensor, 
                                        targets: torch.Tensor, 
                                        input_lens: torch.Tensor,
                                        mask: torch.Tensor) -> float:
        """
        Compute sequence-level accuracy using the simpler approach.
        
        Args:
            logits: (B, T-1, vocab_size) model predictions
            targets: (B, T-1) shifted target tokens
            input_lens: (B,) input lengths (position of "=")
            mask: (B, T-1) mask for output tokens
            
        Returns:
            sequence_accuracy: float between 0.0 and 1.0
        """
        batch_size = targets.shape[0]
        correct_sequences = 0
        
        predictions = logits.argmax(dim=-1)  # (B, T-1)
        
        for i in range(batch_size):
            output_mask = mask[i]  # (T-1,)
            if not output_mask.any():
                continue
                
            # Get predictions and targets for output portion (including padding)
            pred_output = predictions[i][output_mask]  # (output_len,)
            target_output = targets[i][output_mask]  # (output_len,)
            
            # Check if entire output sequence matches (including padding)
            if torch.equal(pred_output, target_output):
                correct_sequences += 1
        
        return correct_sequences / batch_size if batch_size > 0 else 0.0
    
    def generate(self,
                input_tokens: torch.Tensor,
                op_id: int,
                input_len: int,
                max_new_tokens: int = 10,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Generate arithmetic solution given input.
        
        Args:
            input_tokens: (1, T) input tokens up to "="
            op_id: Operation ID
            input_len: Length until "="
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            generated_tokens: (1, T + generated_len) complete sequence
        """
        self.eval()
        device = input_tokens.device
        
        # Ensure we have the "=" and space tokens
        if input_tokens[0, -2:].tolist() != [self.tokenizer.EQUALS_TOKEN, self.tokenizer.SPACE_TOKEN]:
            # Add "=" and space if not present
            equals_space = torch.tensor([[self.tokenizer.EQUALS_TOKEN, self.tokenizer.SPACE_TOKEN]], 
                                      device=device)
            input_tokens = torch.cat([input_tokens, equals_space], dim=1)
        
        generated = input_tokens.clone()
        op_ids = torch.tensor([op_id], device=device)
        input_lens = torch.tensor([input_len], device=device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass (without loss computation for generation)
                logits = self.forward(generated, op_ids, input_lens, return_loss=False)  # (1, T, vocab_size)
                
                # Get logits for next token
                next_token_logits = logits[0, -1, :] / temperature  # (vocab_size,)
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)  # (1,)
                
                # Stop if we generate padding token
                if next_token.item() == self.tokenizer.PAD_TOKEN:
                    break
                
                # Append to sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        return generated
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 