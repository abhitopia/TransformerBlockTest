# Arithmetic Dataset for TransComputer

A scalable, compilation-friendly arithmetic dataset for training TransComputer models on mathematical reasoning tasks.

## 🚀 Key Features

### ✅ **Scalability** 
- **Sampling-based generation** instead of exhaustive enumeration
- Scales to any `max_digits` (tested up to 10+ digits)
- Memory efficient: O(samples) instead of O(10^(2*max_digits))

### ✅ **No Train/Val Overlap**
- **Hash-based deterministic splitting** ensures zero overlap
- Uses SHA256 hash of (operand1, operand2, operation) for consistent assignment
- Reproducible splits with configurable train/val ratios

### ✅ **Compilation Ready**
- **Fixed sequence lengths** via bucketing (e.g., [12, 16, 20, 24])
- **Fixed batch sizes** for consistent tensor shapes
- Perfect for `torch.compile()` optimization

### ✅ **PyTorch Integration**
- Proper `torch.utils.data.Dataset` implementation
- Custom `DataLoader` with collate functions
- Configurable batch sizes, shuffling, and multi-processing

## 📊 Dataset Format

### Token Mapping
- **Digits 0-9** → Token IDs 0-9 (easy interpretation)
- **Special tokens**:
  - `EQUALS_TOKEN = 10` (for "=")
  - `SPACE_TOKEN = 11` (for spaces)  
  - `PAD_TOKEN = 12` (for padding)

### Example Sequence
```
Expression: "89 76 = 6764"
Token IDs:  [8, 9, 11, 7, 6, 11, 10, 11, 6, 7, 6, 4]
Input len:  6  # Length until "=" (for masking)
Op ID:      2  # MUL operation
```

### Output Format
Each batch contains:
- `token_ids`: (B, T) padded sequences
- `op_ids`: (B,) operation IDs for TransComputer programs  
- `input_lens`: (B,) lengths until "=" for cross-attention masking

## 🔧 Usage

### Basic Usage (Recommended for Training)
```python
from arithmatic_dataset import ArithmeticDatasetGenerator, BinaryOp, create_unified_dataloader

# Create generator
generator = ArithmeticDatasetGenerator(
    max_digits=5,  # Scales to any size!
    operations=[BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL]
)

# Generate datasets with no overlap (sequence lengths auto-detected!)
train_datasets, val_datasets = generator.create_datasets(
    train_samples=50000,     # Number of training samples
    val_samples=10000,       # Number of validation samples
    train_ratio=0.8,         # Hash-based splitting ratio
    auto_detect_lengths=True,  # Automatic detection (default)
    seed=42
)

# Create unified DataLoaders (RECOMMENDED for training)
train_loader = create_unified_dataloader(train_datasets, batch_size=32, shuffle=True)
val_loader = create_unified_dataloader(val_datasets, batch_size=32, shuffle=False)
```

### Manual Sequence Length Override (Optional)
```python
# If you need specific sequence lengths for some reason
train_datasets, val_datasets = generator.create_datasets(
    train_samples=50000,
    val_samples=10000,
    fixed_seq_lengths=[16, 24, 32, 40],  # Manual override
    auto_detect_lengths=False,           # Disable auto-detection
    seed=42
)
```

### Training with TransComputer
```python
from TransComputer import TransComputer, Config

# Configure model
config = Config(
    n_layers=6,
    d_model=128,
    n_heads=8,
    n_programs=3,  # ADD=0, SUB=1, MUL=2
    n_prog_tokens=16
)

model = TransComputer(config)
embedding = nn.Embedding(generator.tokenizer.vocab_size, config.d_model)
lm_head = nn.Linear(config.d_model, generator.tokenizer.vocab_size)

# Training loop with unified DataLoader
for epoch in range(num_epochs):
    for batch_dict in train_loader:  # Properly shuffled across all sequence lengths!
        total_loss = 0.0
        
        # Each batch_dict contains samples grouped by sequence length
        for seq_len, batch_data in batch_dict.items():
            # Extract data for this sequence length group
            token_ids = batch_data['token_ids']    # (B_seq, T)
            op_ids = batch_data['op_ids']          # (B_seq,)
            input_lens = batch_data['input_lens']  # (B_seq,)
            
            # Forward pass
            x = embedding(token_ids)
            output = model.forward(
                x=x,
                prog_ids=op_ids,
                input_lens=input_lens,
                compute_steps=3
            )
            
            # Compute loss for this sequence length group
            logits = lm_head(output)
            loss = compute_loss(logits, token_ids)
            total_loss += loss
        
        # Backprop on combined loss from all sequence lengths
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Alternative: Separate DataLoaders
```python
from arithmatic_dataset import create_dataloaders

# For evaluation or specific sequence length analysis
train_loaders = create_dataloaders(train_datasets, batch_size=32, shuffle=True)
val_loaders = create_dataloaders(val_datasets, batch_size=32, shuffle=False)

# Process each sequence length separately
for seq_len, loader in train_loaders.items():
    for batch in loader:
        # Process single sequence length
        pass
```

## 🤖 Automatic Sequence Length Detection

### How It Works
The dataset automatically analyzes a sample of your data to determine optimal sequence length buckets:

1. **Sampling**: Generates 1000 samples (or 10% of total) to analyze length distribution
2. **Analysis**: Finds min/max lengths and unique length values
3. **Bucketing**: Creates 3-5 optimal buckets that cover the range efficiently
4. **Alignment**: Rounds bucket sizes to multiples of 4 for better memory performance

### Example Output
```
Auto-detecting sequence length buckets from 1000 samples...
  Sequence length range: 7 to 16
  Unique lengths found: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
  Auto-detected buckets: [12, 16, 20]
```

### Benefits
- **Zero Configuration**: No need to manually figure out sequence lengths
- **Optimal Performance**: Buckets sized based on actual data distribution
- **Memory Efficient**: Aligned bucket sizes for better GPU memory usage
- **Adaptive**: Works with any combination of max_digits and operations

### Manual Override
If you need specific bucket sizes (e.g., for benchmarking), you can still override:
```python
train_datasets, val_datasets = generator.create_datasets(
    fixed_seq_lengths=[16, 32, 64],  # Custom buckets
    auto_detect_lengths=False        # Disable auto-detection
)
```

## 🔄 DataLoader Approaches

### 🎯 Unified DataLoader (Recommended for Training)
- **Function**: `create_unified_dataloader()`
- **Purpose**: Single DataLoader that properly shuffles across all sequence lengths
- **Output**: Each batch contains samples grouped by sequence length
- **Benefits**: 
  - Proper shuffling across the entire dataset
  - Maintains compilation efficiency (fixed shapes within groups)
  - Simplified training loop
  - Automatic load balancing across sequence lengths

```python
# Single unified loader
train_loader = create_unified_dataloader(train_datasets, batch_size=32, shuffle=True)

for batch_dict in train_loader:
    # batch_dict = {seq_len: batch_data, ...}
    for seq_len, batch_data in batch_dict.items():
        # Process each sequence length group
        pass
```

### 🔧 Separate DataLoaders (Alternative)
- **Function**: `create_dataloaders()`
- **Purpose**: Individual DataLoaders for each sequence length
- **Output**: Dictionary mapping sequence_length → DataLoader
- **Use Cases**:
  - Evaluation/analysis of specific sequence lengths
  - Custom training schedules per sequence length
  - Debugging and profiling

```python
# Multiple separate loaders
train_loaders = create_dataloaders(train_datasets, batch_size=32, shuffle=True)

for seq_len, loader in train_loaders.items():
    for batch in loader:
        # Process single sequence length
        pass
```

## 📈 Scalability Comparison

| Approach | max_digits=2 | max_digits=5 | max_digits=10 |
|----------|--------------|--------------|---------------|
| **Exhaustive** | 9,801 samples | 9.8B samples | 10^20 samples |
| **Sampling** | Configurable | Configurable | Configurable |

## 🎯 Operations Supported

- **Addition** (`BinaryOp.ADD`): `a + b = c`
- **Subtraction** (`BinaryOp.SUB`): `a - b = c` (ensures c ≥ 0)
- **Multiplication** (`BinaryOp.MUL`): `a × b = c`

## 🔍 Sequence Length Buckets

Automatic bucketing based on `max_digits`:

| max_digits | Default Buckets |
|------------|-----------------|
| ≤ 1 | [8, 12] |
| ≤ 2 | [12, 16, 20] |
| ≤ 5 | [16, 24, 32, 40] |
| > 5 | [24, 32, 48, 64] |

## 🧪 Testing

Run the examples:
```bash
# Test basic functionality
python arithmatic_dataset.py

# Test TransComputer integration  
python train_example.py
```

## 🔧 Configuration Options

### ArithmeticDatasetGenerator
- `max_digits`: Maximum digits per operand (scales to any size)
- `operations`: List of operations to include

### create_datasets()
- `train_samples`: Number of training samples to generate
- `val_samples`: Number of validation samples to generate  
- `train_ratio`: Hash-based splitting ratio (ensures no overlap)
- `auto_detect_lengths`: Automatic detection of sequence lengths (default)
- `seed`: Random seed for reproducible generation

### create_dataloaders()
- `batch_size`: Fixed batch size for compilation
- `shuffle`: Whether to shuffle training data
- `num_workers`: Number of DataLoader workers

## 🎯 Perfect for TransComputer

- **Operation IDs** map directly to TransComputer programs
- **Input length masking** works with `get_key_padding_mask()`
- **Fixed tensor shapes** enable efficient compilation
- **Scalable generation** supports any problem complexity 