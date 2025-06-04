"""
Evaluation script for trained ArithmeticModel checkpoints.
Tests the model on fresh arithmetic problems and shows detailed examples.
"""

import torch
import torch.nn as nn
import json
import os
import argparse
from typing import Dict, List, Tuple
import random

from arithmetic_model import ArithmeticModel, ArithmeticModelConfig
from arithmatic_dataset import (
    ArithmeticDatasetGenerator, 
    BinaryOp, 
    create_unified_dataloader,
    ArithmeticTokenizer
)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[ArithmeticModel, Dict]:
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load configs
    model_config = ArithmeticModelConfig(**checkpoint['model_config'])
    training_config = checkpoint['training_config']
    
    # Create and load model
    model = ArithmeticModel(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Loaded model with {model.get_num_params():,} parameters")
    print(f"✅ Training step: {checkpoint['step']}")
    print(f"✅ Best validation accuracy: {checkpoint.get('best_val_accuracy', 'N/A')}")
    
    return model, checkpoint


def evaluate_model(
    model: ArithmeticModel,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_examples: int = 20,
    max_eval_batches: int = 100
) -> Dict[str, float]:
    """Evaluate model and show examples"""
    model.eval()
    
    total_loss = 0.0
    total_sequence_accuracy = 0.0
    total_token_accuracy = 0.0
    total_sequences = 0
    total_tokens = 0
    num_batches = 0
    
    # For collecting examples
    examples = []
    tokenizer = ArithmeticTokenizer()
    
    # Operation names and symbols for display
    op_names = {0: "ADD", 1: "SUB", 2: "MUL"}
    op_symbols = {0: "+", 1: "-", 2: "*"}
    
    print(f"\n🧮 Evaluating model on {max_eval_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(eval_loader):
            if batch_idx >= max_eval_batches:
                break
            
            # Move to device
            device_batch = {}
            for seq_len, batch_data in batch_dict.items():
                device_batch[seq_len] = {
                    'token_ids': batch_data['token_ids'].to(device),
                    'op_ids': batch_data['op_ids'].to(device),
                    'input_lens': batch_data['input_lens'].to(device),
                    'batch_size': batch_data['batch_size']
                }
            
            # Process each sequence length group
            for seq_len, batch_data in device_batch.items():
                # Forward pass
                loss, logits, mask = model.forward(
                    token_ids=batch_data['token_ids'],
                    op_ids=batch_data['op_ids'],
                    input_lens=batch_data['input_lens'],
                    return_loss=True
                )
                
                # Compute metrics
                targets = batch_data['token_ids'][:, 1:]
                metrics = model.compute_metrics(loss, logits, targets, batch_data['input_lens'], mask)
                
                batch_size = batch_data['batch_size']
                total_loss += loss.item()
                total_sequence_accuracy += metrics['sequence_accuracy'] * batch_size
                total_token_accuracy += metrics['token_accuracy'] * metrics['num_output_tokens']
                total_sequences += batch_size
                total_tokens += metrics['num_output_tokens']
                
                # Collect examples for display
                if len(examples) < num_examples:
                    predictions = torch.argmax(logits, dim=-1)  # (B, T-1)
                    
                    token_ids = batch_data['token_ids']
                    op_ids = batch_data['op_ids']
                    input_lens = batch_data['input_lens']
                    
                    for i in range(min(batch_size, num_examples - len(examples))):
                        input_len = input_lens[i].item()
                        op_id = op_ids[i].item()
                        
                        # Parse operands
                        operands = extract_operands_from_sequence(token_ids[i], input_len, tokenizer)
                        
                        # Get operation info
                        op_name = op_names.get(op_id, f"OP{op_id}")
                        op_symbol = op_symbols.get(op_id, "?")
                        
                        # Get target
                        target_tokens = token_ids[i].cpu().tolist()
                        target_str = tokenizer.detokenize(target_tokens)
                        
                        # Get prediction (corrected logic)
                        predicted_tokens = token_ids[i, :input_len+1].cpu().tolist()  # Include "=" token
                        
                        # Add the predicted output tokens (starting from space after "=")
                        output_start_pred_idx = input_len  # Index in predictions array for space after "="
                        predicted_output = predictions[i, output_start_pred_idx:].cpu().tolist()
                        predicted_tokens.extend(predicted_output)
                        
                        # Ensure same length as target
                        target_len = len(target_tokens)
                        if len(predicted_tokens) < target_len:
                            predicted_tokens.extend([tokenizer.PAD_TOKEN] * (target_len - len(predicted_tokens)))
                        elif len(predicted_tokens) > target_len:
                            predicted_tokens = predicted_tokens[:target_len]
                        
                        predicted_str = tokenizer.detokenize(predicted_tokens)
                        
                        # Check if prediction is correct
                        is_correct = (target_str.strip() == predicted_str.strip())
                        status = "✅" if is_correct else "❌"
                        
                        # Format example
                        if operands and len(operands) >= 2:
                            operation_desc = f"{operands[0]} {op_symbol} {operands[1]}"
                        else:
                            operation_desc = f"[{op_name}]"
                        
                        examples.append({
                            'operation': operation_desc,
                            'target': target_str,
                            'predicted': predicted_str,
                            'correct': is_correct,
                            'status': status
                        })
            
            num_batches += 1
    
    # Compute final metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_sequence_accuracy = total_sequence_accuracy / total_sequences if total_sequences > 0 else 0.0
    avg_token_accuracy = total_token_accuracy / total_tokens if total_tokens > 0 else 0.0
    
    results = {
        'loss': avg_loss,
        'sequence_accuracy': avg_sequence_accuracy,
        'token_accuracy': avg_token_accuracy,
        'total_sequences': total_sequences,
        'total_tokens': total_tokens,
        'examples': examples
    }
    
    return results


def extract_operands_from_sequence(token_ids: torch.Tensor, input_len: int, tokenizer: ArithmeticTokenizer) -> List[str]:
    """Extract operands from input portion of sequence"""
    try:
        input_tokens = token_ids[:input_len].cpu().tolist()
        input_str = tokenizer.detokenize(input_tokens)
        parts = [part for part in input_str.split(' ') if part.strip()]
        if len(parts) >= 2:
            return [parts[0], parts[1]]
        return []
    except:
        return []


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained ArithmeticModel checkpoint")
    parser.add_argument("checkpoint_path", help="Path to model checkpoint (.pt file)")
    parser.add_argument("--num-examples", type=int, default=30, help="Number of examples to show")
    parser.add_argument("--eval-samples", type=int, default=5000, help="Number of evaluation samples")
    parser.add_argument("--max-digits", type=int, default=2, help="Maximum digits for test problems")
    parser.add_argument("--operations", nargs="+", default=["ADD", "SUB", "MUL"], help="Operations to test")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    model, checkpoint = load_checkpoint(args.checkpoint_path, device)
    
    # Create evaluation dataset (fresh problems, not from training)
    print(f"\n📊 Generating {args.eval_samples} fresh evaluation samples...")
    
    op_mapping = {"ADD": BinaryOp.ADD, "SUB": BinaryOp.SUB, "MUL": BinaryOp.MUL}
    operations = [op_mapping[op] for op in args.operations]
    
    generator = ArithmeticDatasetGenerator(
        max_digits=args.max_digits,
        operations=operations
    )
    
    # Generate evaluation data (using different seed to ensure fresh problems)
    eval_datasets, _ = generator.create_datasets(
        train_samples=args.eval_samples,
        val_samples=0,  # We don't need validation for evaluation
        auto_detect_lengths=False,
        use_cache=False,  # Don't use cache to ensure fresh problems
        seed=12345  # Different seed from training
    )
    
    eval_loader = create_unified_dataloader(
        eval_datasets,
        batch_size=32,
        shuffle=False
    )
    
    # Evaluate model
    results = evaluate_model(model, eval_loader, device, args.num_examples, max_eval_batches=200)
    
    # Print results
    print(f"\n" + "="*60)
    print(f"📈 EVALUATION RESULTS")
    print(f"="*60)
    print(f"Loss: {results['loss']:.4f}")
    print(f"Sequence Accuracy: {results['sequence_accuracy']:.1%}")
    print(f"Token Accuracy: {results['token_accuracy']:.1%}")
    print(f"Total Sequences: {results['total_sequences']:,}")
    print(f"Total Tokens: {results['total_tokens']:,}")
    
    # Show examples
    print(f"\n🔍 SAMPLE PREDICTIONS:")
    print(f"-" * 80)
    
    # Group examples by correctness
    correct_examples = [ex for ex in results['examples'] if ex['correct']]
    incorrect_examples = [ex for ex in results['examples'] if not ex['correct']]
    
    print(f"\n✅ CORRECT PREDICTIONS ({len(correct_examples)}/{len(results['examples'])}):")
    for i, ex in enumerate(correct_examples[:15]):  # Show first 15 correct
        print(f"  {ex['status']} {ex['operation']}: {ex['predicted']}")
    
    print(f"\n❌ INCORRECT PREDICTIONS:")
    for i, ex in enumerate(incorrect_examples[:15]):  # Show first 15 incorrect
        print(f"  {ex['status']} {ex['operation']}")
        print(f"      Target:    {ex['target']}")
        print(f"      Predicted: {ex['predicted']}")
    
    print(f"\n" + "="*60)
    accuracy_pct = results['sequence_accuracy'] * 100
    print(f"🎯 FINAL SCORE: {accuracy_pct:.1f}% sequence accuracy")
    print(f"="*60)


if __name__ == "__main__":
    main() 