import torch
import random
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
from torch.utils.data import Dataset, DataLoader
import itertools
import hashlib


class BinaryOp(Enum):
    ADD = 0
    SUB = 1
    MUL = 2
    # DIV = 3  # Can be added later if needed


@dataclass
class ArithmeticSample:
    """Single arithmetic sample containing all necessary information"""
    op_id: int
    operand1: int
    operand2: int
    result: int
    token_ids: torch.Tensor  # Tokenized sequence
    input_len: int  # Length until "=" (exclusive)


class ArithmeticTokenizer:
    """Simple tokenizer for arithmetic expressions"""
    
    def __init__(self):
        # Digit tokens (0-9 for digits 0-9)
        self.digit_to_token = {str(i): i for i in range(10)}
        self.token_to_digit = {v: k for k, v in self.digit_to_token.items()}
        
        # Special tokens at the end
        self.EQUALS_TOKEN = 10
        self.SPACE_TOKEN = 11
        self.PAD_TOKEN = 12
        
        # Create vocabulary
        self.vocab_size = 13  # digits 0-9, EQUALS, SPACE, PAD
        
    def tokenize_number(self, num: int) -> List[int]:
        """Convert a number to list of digit tokens"""
        return [self.digit_to_token[digit] for digit in str(num)]
    
    def tokenize_sequence(self, operand1: int, operand2: int, result: int) -> Tuple[List[int], int]:
        """
        Tokenize arithmetic sequence: "operand1 operand2 = result"
        Returns: (token_ids, input_len)
        """
        tokens = []
        
        # Add first operand
        tokens.extend(self.tokenize_number(operand1))
        tokens.append(self.SPACE_TOKEN)
        
        # Add second operand
        tokens.extend(self.tokenize_number(operand2))
        tokens.append(self.SPACE_TOKEN)
        
        # Mark input length (before equals sign)
        input_len = len(tokens)
        
        # Add equals sign
        tokens.append(self.EQUALS_TOKEN)
        tokens.append(self.SPACE_TOKEN)
        
        # Add result
        tokens.extend(self.tokenize_number(result))
        
        return tokens, input_len
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to string for debugging"""
        result = []
        for token_id in token_ids:
            if token_id == self.PAD_TOKEN:
                result.append('_')
            elif token_id == self.EQUALS_TOKEN:
                result.append('=')
            elif token_id == self.SPACE_TOKEN:
                result.append(' ')
            elif token_id in self.token_to_digit:
                result.append(self.token_to_digit[token_id])
            else:
                result.append(f'<UNK:{token_id}>')
        return ''.join(result)


class ArithmeticDataset(Dataset):
    """PyTorch Dataset for arithmetic problems with fixed sequence lengths"""
    
    def __init__(self, samples: List[ArithmeticSample], max_seq_len: int):
        """
        Args:
            samples: List of pre-generated arithmetic samples
            max_seq_len: Fixed sequence length (with padding)
        """
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.tokenizer = ArithmeticTokenizer()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Pad sequence to fixed length
        token_ids = torch.full((self.max_seq_len,), self.tokenizer.PAD_TOKEN, dtype=torch.long)
        seq_len = min(len(sample.token_ids), self.max_seq_len)
        token_ids[:seq_len] = sample.token_ids[:seq_len]
        
        return {
            'token_ids': token_ids,
            'op_id': torch.tensor(sample.op_id, dtype=torch.long),
            'input_len': torch.tensor(sample.input_len, dtype=torch.long),
            'operand1': sample.operand1,
            'operand2': sample.operand2,
            'result': sample.result
        }


class ArithmeticDatasetGenerator:
    """Generator for creating train/val splits with no overlap using sampling"""
    
    def __init__(self, max_digits: int = 2, operations: List[BinaryOp] = None):
        """
        Args:
            max_digits: Maximum number of digits for operands
            operations: List of binary operations to include
        """
        self.max_digits = max_digits
        self.operations = operations or [BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL]
        self.tokenizer = ArithmeticTokenizer()
        
        # Calculate operand ranges
        self.min_operand = 1  # Avoid 0 to prevent trivial cases
        self.max_operand = 10 ** max_digits - 1
        
        # Calculate theoretical max sequence length
        self.max_seq_len = self.calculate_max_seq_len()
        
        # Calculate total possible combinations
        self.total_combinations = self._calculate_total_combinations()
    
    def calculate_max_seq_len(self) -> int:
        """
        Calculate theoretical maximum sequence length based on max_digits and operations.
        
        Format: "operand1 operand2 = result"
        Example: "99 99 = 9801" (for 2-digit operands)
        """
        # Operand lengths
        operand1_len = self.max_digits  # e.g., "99" = 2 tokens
        operand2_len = self.max_digits  # e.g., "99" = 2 tokens
        
        # Calculate maximum result length for each operation
        max_result_lens = []
        
        for op in self.operations:
            if op == BinaryOp.ADD:
                # Max: 999 + 999 = 1998 (max_digits + 1)
                max_result_len = self.max_digits + 1
            elif op == BinaryOp.SUB:
                # Max: 999 - 1 = 998 (max_digits)
                max_result_len = self.max_digits
            elif op == BinaryOp.MUL:
                # Max: 999 * 999 = 998001 (2 * max_digits)
                max_result_len = 2 * self.max_digits
            else:
                # Default fallback
                max_result_len = 2 * self.max_digits
                
            max_result_lens.append(max_result_len)
        
        # Get the maximum across all operations
        max_result_len = max(max_result_lens)
        
        # Calculate total sequence length:
        # operand1 + space + operand2 + space + equals + space + result
        total_len = (
            operand1_len +      # e.g., "99"
            1 +                 # space
            operand2_len +      # e.g., "99" 
            1 +                 # space
            1 +                 # equals sign
            1 +                 # space after equals
            max_result_len      # result (e.g., "9801")
        )
        
        # Add minimal padding for bucketing efficiency (round up to multiple of 4)
        return ((total_len + 3) // 4) * 4
    
    def _calculate_total_combinations(self) -> int:
        """
        Calculate total number of possible combinations.
        
        Returns:
            Total number of unique (operand1, operand2, operation) combinations
        """
        # Number of possible operand pairs
        num_operand_pairs = (self.max_operand - self.min_operand + 1) ** 2
        
        # Multiply by number of operations
        return num_operand_pairs * len(self.operations)
    
    def _validate_sample_sizes(self, train_samples: int, val_samples: int) -> None:
        """
        Validate that sample sizes are reasonable compared to total combinations.
        
        Args:
            train_samples: Number of training samples
            val_samples: Number of validation samples
        """
        total_samples = train_samples + val_samples
        coverage_ratio = total_samples / self.total_combinations
        
        print(f"\n=== Sample Size Validation ===")
        print(f"Max digits: {self.max_digits}")
        print(f"Operations: {[op.name for op in self.operations]}")
        print(f"Operand range: {self.min_operand} to {self.max_operand}")
        print(f"Total possible combinations: {self.total_combinations:,}")
        print(f"Training samples: {train_samples:,}")
        print(f"Validation samples: {val_samples:,}")
        print(f"Total samples: {total_samples:,}")
        print(f"Coverage ratio: {coverage_ratio:.4f} ({coverage_ratio*100:.2f}%)")
        print(f"Theoretical max sequence length: {self.max_seq_len} tokens")
        
        # Warnings and recommendations
        if coverage_ratio > 0.5:
            print("⚠️  WARNING: Using >50% of possible combinations - high risk of overfitting!")
            print("   Consider reducing sample size or increasing max_digits")
        elif coverage_ratio > 0.1:
            print("⚠️  WARNING: Using >10% of possible combinations - moderate risk of overfitting")
            print("   Consider monitoring validation performance carefully")
        elif coverage_ratio < 0.001:
            print("ℹ️  INFO: Using <0.1% of combinations - very low overfitting risk")
        else:
            print("✅ Good: Using reasonable fraction of combinations")
        
        # Check if we have enough combinations
        min_recommended = total_samples * 10  # At least 10x more combinations than samples
        if self.total_combinations < min_recommended:
            print(f"⚠️  WARNING: Consider increasing max_digits to get more combinations")
            print(f"   Recommended: at least {min_recommended:,} combinations")
        
        print("=" * 30)
    
    def _compute_result(self, op1: int, op2: int, operation: BinaryOp) -> Optional[int]:
        """Compute result of binary operation"""
        if operation == BinaryOp.ADD:
            return op1 + op2
        elif operation == BinaryOp.SUB:
            # Ensure non-negative results
            if op1 >= op2:
                return op1 - op2
            else:
                return op2 - op1
        elif operation == BinaryOp.MUL:
            return op1 * op2
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _hash_sample(self, op1: int, op2: int, operation: BinaryOp) -> str:
        """Create a deterministic hash for a sample to ensure train/val separation"""
        # Create a unique string representation
        sample_str = f"{op1}_{op2}_{operation.value}"
        # Use SHA256 for deterministic hashing
        return hashlib.sha256(sample_str.encode()).hexdigest()
    
    def _is_train_sample(self, op1: int, op2: int, operation: BinaryOp, train_ratio: float) -> bool:
        """Deterministically decide if a sample belongs to train or val set"""
        hash_str = self._hash_sample(op1, op2, operation)
        # Use first 8 characters of hash to get a number between 0 and 1
        hash_int = int(hash_str[:8], 16)
        hash_ratio = hash_int / (16**8 - 1)  # Normalize to [0, 1]
        return hash_ratio < train_ratio
    
    def _generate_random_sample(self, rng: random.Random) -> ArithmeticSample:
        """Generate a single random arithmetic sample"""
        # Choose random operation
        operation = rng.choice(self.operations)
        
        # Generate operands
        op1 = rng.randint(self.min_operand, self.max_operand)
        op2 = rng.randint(self.min_operand, self.max_operand)
        
        # For subtraction, ensure positive results
        if operation == BinaryOp.SUB and op1 < op2:
            op1, op2 = op2, op1
        
        # Compute result
        result = self._compute_result(op1, op2, operation)
        
        # Tokenize sequence
        token_ids, input_len = self.tokenizer.tokenize_sequence(op1, op2, result)
        
        return ArithmeticSample(
            op_id=operation.value,
            operand1=op1,
            operand2=op2,
            result=result,
            token_ids=torch.tensor(token_ids, dtype=torch.long),
            input_len=input_len
        )
    
    def _generate_samples_for_split(self, 
                                   num_samples: int, 
                                   is_train: bool, 
                                   train_ratio: float,
                                   seed: int) -> List[ArithmeticSample]:
        """Generate samples for either train or val split"""
        samples = []
        rng = random.Random(seed)
        attempts = 0
        max_attempts = num_samples * 10  # Prevent infinite loops
        
        while len(samples) < num_samples and attempts < max_attempts:
            sample = self._generate_random_sample(rng)
            
            # Check if this sample belongs to the desired split
            sample_is_train = self._is_train_sample(
                sample.operand1, sample.operand2, 
                BinaryOp(sample.op_id), train_ratio
            )
            
            if sample_is_train == is_train:
                # Check for duplicates (though very unlikely with large spaces)
                sample_key = (sample.operand1, sample.operand2, sample.op_id)
                if not any(s.operand1 == sample.operand1 and 
                          s.operand2 == sample.operand2 and 
                          s.op_id == sample.op_id for s in samples):
                    samples.append(sample)
            
            attempts += 1
        
        if len(samples) < num_samples:
            print(f"Warning: Only generated {len(samples)}/{num_samples} samples for {'train' if is_train else 'val'}")
        
        return samples
    
    def _auto_detect_sequence_lengths(self, sample_size: int = 1000, seed: int = 42) -> List[int]:
        """
        Automatically detect optimal sequence length buckets by sampling data.
        
        Args:
            sample_size: Number of samples to generate for length analysis
            seed: Random seed for reproducible sampling
            
        Returns:
            List of optimal sequence length buckets
        """
        print(f"Auto-detecting sequence length buckets from {sample_size} samples...")
        
        # Generate sample data to analyze sequence lengths
        rng = random.Random(seed)
        sequence_lengths = []
        
        for _ in range(sample_size):
            sample = self._generate_random_sample(rng)
            sequence_lengths.append(len(sample.token_ids))
        
        # Analyze the distribution
        min_len = min(sequence_lengths)
        max_len = max(sequence_lengths)
        unique_lengths = sorted(set(sequence_lengths))
        
        print(f"  Sequence length range: {min_len} to {max_len}")
        print(f"  Unique lengths found: {unique_lengths}")
        
        # Create buckets that cover the range efficiently
        if len(unique_lengths) <= 4:
            # If we have few unique lengths, use them directly with some padding
            buckets = [length + 2 for length in unique_lengths]  # Add 2 for safety margin
        else:
            # Create 3-5 buckets that cover the range
            num_buckets = min(5, max(3, len(unique_lengths) // 3))
            bucket_size = (max_len - min_len) // num_buckets + 1
            
            buckets = []
            for i in range(num_buckets):
                bucket_len = min_len + (i + 1) * bucket_size
                # Round up to nearest multiple of 4 for better memory alignment
                bucket_len = ((bucket_len + 3) // 4) * 4
                buckets.append(bucket_len)
            
            # Ensure the last bucket covers the maximum length
            if buckets[-1] < max_len:
                buckets[-1] = ((max_len + 3) // 4) * 4
        
        # Remove duplicates and sort
        buckets = sorted(list(set(buckets)))
        
        print(f"  Auto-detected buckets: {buckets}")
        return buckets

    def create_datasets(self, 
                       train_samples: int = 10000,
                       val_samples: int = 2000,
                       train_ratio: float = 0.8,
                       fixed_seq_lengths: Optional[List[int]] = None,
                       auto_detect_lengths: bool = False,
                       seed: int = 42) -> Tuple[Dict[int, ArithmeticDataset], Dict[int, ArithmeticDataset]]:
        """
        Create train/val datasets with no overlap using sampling
        
        Args:
            train_samples: Number of training samples to generate
            val_samples: Number of validation samples to generate
            train_ratio: Ratio used for hash-based splitting (ensures no overlap)
            fixed_seq_lengths: List of fixed sequence lengths for bucketing (optional)
            auto_detect_lengths: Whether to automatically detect optimal sequence lengths (deprecated)
            seed: Random seed for reproducible generation
            
        Returns:
            Tuple of (train_datasets_dict, val_datasets_dict) where keys are sequence lengths
        """
        # Validate sample sizes vs total combinations
        self._validate_sample_sizes(train_samples, val_samples)
        
        if fixed_seq_lengths is None:
            # Use theoretical max sequence length (much more principled than auto-detection)
            fixed_seq_lengths = [self.max_seq_len]
            print(f"Using theoretical max sequence length: {self.max_seq_len} tokens")
            print(f"  (Based on max_digits={self.max_digits} and operations={[op.name for op in self.operations]})")
        else:
            print(f"Using provided sequence length buckets: {fixed_seq_lengths}")
            # Warn if provided lengths might be too small
            if max(fixed_seq_lengths) < self.max_seq_len:
                print(f"⚠️  WARNING: Max bucket size ({max(fixed_seq_lengths)}) < theoretical max ({self.max_seq_len})")
                print("   Some sequences might be truncated!")
        
        print(f"Generating {train_samples} train and {val_samples} val samples...")
        
        # Generate train and val samples separately
        train_samples_list = self._generate_samples_for_split(
            train_samples, is_train=True, train_ratio=train_ratio, seed=seed
        )
        val_samples_list = self._generate_samples_for_split(
            val_samples, is_train=False, train_ratio=train_ratio, seed=seed + 1
        )
        
        # Group samples by sequence length
        def group_by_length(samples):
            samples_by_length = {}
            for sample in samples:
                seq_len = len(sample.token_ids)
                # Find the appropriate bucket
                bucket_len = min([l for l in fixed_seq_lengths if l >= seq_len], 
                                default=max(fixed_seq_lengths))
                
                if bucket_len not in samples_by_length:
                    samples_by_length[bucket_len] = []
                samples_by_length[bucket_len].append(sample)
            return samples_by_length
        
        train_by_length = group_by_length(train_samples_list)
        val_by_length = group_by_length(val_samples_list)
        
        # Create datasets
        train_datasets = {}
        val_datasets = {}
        
        all_seq_lens = set(train_by_length.keys()) | set(val_by_length.keys())
        
        for seq_len in all_seq_lens:
            train_samples_for_len = train_by_length.get(seq_len, [])
            val_samples_for_len = val_by_length.get(seq_len, [])
            
            if train_samples_for_len:
                train_datasets[seq_len] = ArithmeticDataset(train_samples_for_len, seq_len)
            if val_samples_for_len:
                val_datasets[seq_len] = ArithmeticDataset(val_samples_for_len, seq_len)
            
            print(f"Sequence length {seq_len}: {len(train_samples_for_len)} train, {len(val_samples_for_len)} val samples")
        
        return train_datasets, val_datasets


def create_dataloaders(datasets: Dict[int, ArithmeticDataset], 
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 0,
                      drop_last: bool = True) -> Dict[int, DataLoader]:
    """Create DataLoaders for each sequence length bucket (for separate processing)"""
    dataloaders = {}
    
    for seq_len, dataset in datasets.items():
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=collate_batch,
            drop_last=drop_last
        )
        dataloaders[seq_len] = dataloader
    
    return dataloaders


class MultiLengthDataset(Dataset):
    """
    Unified dataset that combines multiple sequence length buckets.
    Each sample includes its sequence length for dynamic batching.
    """
    
    def __init__(self, datasets: Dict[int, ArithmeticDataset]):
        """
        Args:
            datasets: Dictionary mapping sequence_length -> ArithmeticDataset
        """
        self.datasets = datasets
        self.seq_lengths = list(datasets.keys())
        
        # Create a flat list of (dataset_idx, sample_idx, seq_len) tuples
        self.sample_indices = []
        for seq_len, dataset in datasets.items():
            for sample_idx in range(len(dataset)):
                self.sample_indices.append((seq_len, sample_idx))
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        seq_len, sample_idx = self.sample_indices[idx]
        sample = self.datasets[seq_len][sample_idx]
        # Add sequence length to the sample
        sample['seq_len'] = torch.tensor(seq_len, dtype=torch.long)
        return sample


def multi_length_collate_batch(batch):
    """
    Custom collate function that groups samples by sequence length
    and creates separate batches for each length.
    
    Returns a dictionary where keys are sequence lengths and values are batched data.
    """
    # Group samples by sequence length
    samples_by_length = {}
    for sample in batch:
        seq_len = sample['seq_len'].item()
        if seq_len not in samples_by_length:
            samples_by_length[seq_len] = []
        samples_by_length[seq_len].append(sample)
    
    # Create batches for each sequence length
    batched_data = {}
    for seq_len, samples in samples_by_length.items():
        if len(samples) > 0:
            token_ids = torch.stack([s['token_ids'] for s in samples])
            op_ids = torch.stack([s['op_id'] for s in samples])
            input_lens = torch.stack([s['input_len'] for s in samples])
            
            batched_data[seq_len] = {
                'token_ids': token_ids,
                'op_ids': op_ids,
                'input_lens': input_lens,
                'batch_size': len(samples)
            }
    
    return batched_data


def create_unified_dataloader(datasets: Dict[int, ArithmeticDataset],
                             batch_size: int = 32,
                             shuffle: bool = True,
                             num_workers: int = 0,
                             drop_last: bool = True) -> DataLoader:
    """
    Create a single unified DataLoader that handles multiple sequence lengths.
    
    This is the recommended approach for training as it provides proper shuffling
    across all sequence lengths while maintaining efficient batching.
    
    Args:
        datasets: Dictionary mapping sequence_length -> ArithmeticDataset
        batch_size: Total batch size (will be distributed across sequence lengths)
        shuffle: Whether to shuffle samples
        num_workers: Number of DataLoader workers
        drop_last: Whether to drop incomplete batches (recommended for compilation)
        
    Returns:
        Single DataLoader that yields batches grouped by sequence length
    """
    unified_dataset = MultiLengthDataset(datasets)
    
    return DataLoader(
        unified_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=multi_length_collate_batch,
        drop_last=drop_last
    )


def collate_batch(batch):
    """Custom collate function for single-length DataLoader"""
    token_ids = torch.stack([item['token_ids'] for item in batch])
    op_ids = torch.stack([item['op_id'] for item in batch])
    input_lens = torch.stack([item['input_len'] for item in batch])
    
    return {
        'token_ids': token_ids,
        'op_ids': op_ids,
        'input_lens': input_lens
    }


# Example usage and testing
if __name__ == "__main__":
    print("=== Automatic Sequence Length Detection Demo ===")
    
    # Test 1: Small digits with auto-detection
    print("\n--- Test 1: max_digits=1 with auto-detection ---")
    generator1 = ArithmeticDatasetGenerator(max_digits=1, operations=[BinaryOp.ADD, BinaryOp.MUL])
    train_datasets1, val_datasets1 = generator1.create_datasets(
        train_samples=500,
        val_samples=100,
        auto_detect_lengths=True,  # Automatic detection (default)
        seed=42
    )
    
    # Test 2: Larger digits with auto-detection
    print("\n--- Test 2: max_digits=3 with auto-detection ---")
    generator2 = ArithmeticDatasetGenerator(max_digits=3, operations=[BinaryOp.ADD, BinaryOp.MUL])
    train_datasets2, val_datasets2 = generator2.create_datasets(
        train_samples=500,
        val_samples=100,
        auto_detect_lengths=True,
        seed=42
    )
    
    # Test 3: Manual override
    print("\n--- Test 3: Manual sequence length override ---")
    generator3 = ArithmeticDatasetGenerator(max_digits=2, operations=[BinaryOp.ADD, BinaryOp.MUL])
    train_datasets3, val_datasets3 = generator3.create_datasets(
        train_samples=500,
        val_samples=100,
        fixed_seq_lengths=[16, 24],  # Manual override
        auto_detect_lengths=False,   # Disable auto-detection
        seed=42
    )
    
    print("\n=== Dataset Statistics ===")
    for i, (train_ds, val_ds) in enumerate([(train_datasets1, val_datasets1), 
                                           (train_datasets2, val_datasets2),
                                           (train_datasets3, val_datasets3)], 1):
        total_train = sum(len(ds) for ds in train_ds.values())
        total_val = sum(len(ds) for ds in val_ds.values())
        print(f"Test {i}: {total_train} train, {total_val} val samples")
        print(f"  Buckets: {list(train_ds.keys())}")
    
    # Test for overlap (should be zero)
    print("\n=== Overlap Test ===")
    train_keys = set()
    val_keys = set()
    
    for dataset in train_datasets1.values():
        for i in range(len(dataset)):
            sample = dataset[i]
            key = (sample['operand1'], sample['operand2'], sample['op_id'].item())
            train_keys.add(key)
    
    for dataset in val_datasets1.values():
        for i in range(len(dataset)):
            sample = dataset[i]
            key = (sample['operand1'], sample['operand2'], sample['op_id'].item())
            val_keys.add(key)
    
    overlap = train_keys & val_keys
    print(f"Train/Val overlap: {len(overlap)} samples (should be 0)")
    
    # ===== RECOMMENDED: Unified DataLoader for Training =====
    print("\n=== Unified DataLoader (Auto-detected buckets) ===")
    unified_train_loader = create_unified_dataloader(train_datasets2, batch_size=8, shuffle=True)
    
    print("Processing unified batches with auto-detected sequence lengths:")
    for i, batch_dict in enumerate(unified_train_loader):
        print(f"\nBatch {i+1}:")
        for seq_len, batch_data in batch_dict.items():
            print(f"  Seq len {seq_len}: {batch_data['batch_size']} samples, shape {batch_data['token_ids'].shape}")
            
            # Show first sample
            tokenizer = ArithmeticTokenizer()
            first_sample = batch_data['token_ids'][0]
            non_pad_tokens = first_sample[first_sample != tokenizer.PAD_TOKEN]
            token_str = tokenizer.detokenize(non_pad_tokens.tolist())
            print(f"    Sample: {token_str}")
        
        if i >= 1:  # Show first 2 batches
            break
    
    print(f"\nTokenizer vocab size: {generator2.tokenizer.vocab_size}")
    print(f"Max digits: {generator2.max_digits} (scales to any size!)")
    
    print("\n=== Key Features ===")
    print("✓ Automatic sequence length bucket detection (default)")
    print("✓ Manual override available if needed")
    print("✓ Optimal bucket sizing based on actual data distribution")
    print("✓ Memory-aligned buckets for better performance")
    print("✓ No user configuration required for sequence lengths")
