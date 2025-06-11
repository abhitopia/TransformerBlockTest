#!/usr/bin/env python3
"""
Associative Attention Head

An attention head implementation that uses heteroassociative memories as the
underlying substrate for Key, Query, and Value operations. Symbols are defined
as tuples of subsymbols, and the head is constructed via rules.

Architecture:
- Phase 1: AssociativeHeadBuilder - Rule construction and matrix building
- Phase 2: AssociativeHead - PyTorch tensor operations

Example usage:
    # Phase 1: Build the head with rules
    builder = AssociativeHeadBuilder(vector_dim=512, composition_method="binding")
    
    # Add linking rules (affects Key and Query memories)
    builder.add_linking_rule(("subject", "person"), ("verb", "action"))
    builder.add_value_rule(("verb", "action"), ("feature", "dynamic"))
    
    # Finalize to get torch module
    head = builder.build()
    
    # Phase 2: Use as standard attention head
    x = torch.randn(batch_size, seq_len, vector_dim)
    output = head(x)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

@dataclass
class VectorInfo:
    """Information about a generated vector."""
    symbol: str
    vector: np.ndarray
    usage_count: int = 0

class OrthogonalVectorFactory:
    """
    Factory for generating nearly orthogonal vectors for associative memory.
    Shared across all memories in the attention head.
    """
    
    def __init__(self, 
                 vector_dim: int = 512,
                 sparsity: float = 1.0,
                 orthogonality_threshold: float = 0.1,
                 max_attempts: int = 100):
        """
        Initialize the vector factory.
        
        Args:
            vector_dim: Dimensionality of vectors
            sparsity: Fraction of non-zero elements (1.0 = dense, 0.1 = sparse)
            orthogonality_threshold: Maximum dot product allowed with existing vectors
            max_attempts: Maximum attempts to generate orthogonal vector
        """
        self.vector_dim = vector_dim
        self.sparsity = sparsity
        self.orthogonality_threshold = orthogonality_threshold
        self.max_attempts = max_attempts
        
        # Storage for issued vectors
        self.vectors: Dict[str, VectorInfo] = {}
        self.vector_matrix: Optional[np.ndarray] = None
        self._needs_matrix_update = False
        
    def _generate_vector(self) -> np.ndarray:
        """Generate a random vector with specified sparsity."""
        vector = np.zeros(self.vector_dim)
        n_nonzero = max(1, int(self.vector_dim * self.sparsity))
        
        # Random positions for non-zero elements
        positions = np.random.choice(self.vector_dim, n_nonzero, replace=False)
        values = np.random.randn(n_nonzero)
        vector[positions] = values
        
        # Normalize to unit length
        return vector / np.linalg.norm(vector)
    
    def _update_vector_matrix(self):
        """Update the matrix of all vectors for efficient orthogonality checking."""
        if not self.vectors or not self._needs_matrix_update:
            return
            
        vectors_list = [info.vector for info in self.vectors.values()]
        self.vector_matrix = np.vstack(vectors_list)
        self._needs_matrix_update = False
    
    def _check_orthogonality(self, candidate_vector: np.ndarray) -> Tuple[bool, float]:
        """Check if candidate vector is sufficiently orthogonal to existing vectors."""
        if not self.vectors:
            return True, 0.0
            
        self._update_vector_matrix()
        similarities = np.abs(self.vector_matrix @ candidate_vector)
        max_similarity = np.max(similarities)
        
        return max_similarity < self.orthogonality_threshold, max_similarity
    
    def get_or_create_vector(self, symbol: str) -> np.ndarray:
        """Get existing vector for symbol or create a new nearly orthogonal one."""
        # Return existing vector if already created
        if symbol in self.vectors:
            self.vectors[symbol].usage_count += 1
            return self.vectors[symbol].vector
        
        # Generate new nearly orthogonal vector
        for attempt in range(self.max_attempts):
            candidate = self._generate_vector()
            is_orthogonal, max_sim = self._check_orthogonality(candidate)
            
            if is_orthogonal:
                self.vectors[symbol] = VectorInfo(
                    symbol=symbol,
                    vector=candidate,
                    usage_count=1
                )
                self._needs_matrix_update = True
                return candidate
        
        # Fallback if orthogonal vector not found
        warnings.warn(f"Could not generate orthogonal vector for '{symbol}' "
                     f"after {self.max_attempts} attempts. Max similarity: {max_sim:.3f}")
        
        self.vectors[symbol] = VectorInfo(
            symbol=symbol,
            vector=candidate,
            usage_count=1
        )
        self._needs_matrix_update = True
        return candidate

class TupleComposer:
    """Composes tuple symbols into single vectors using various methods."""
    
    def __init__(self, 
                 vector_factory: OrthogonalVectorFactory,
                 method: str = "binding"):
        """
        Initialize tuple composer.
        
        Args:
            vector_factory: Factory for generating orthogonal vectors
            method: Composition method ("addition", "multiplication", "binding", "position_encoding")
        """
        self.factory = vector_factory
        self.method = method
        self.subsymbol_cache: Dict[str, np.ndarray] = {}
    
    def _get_subsymbol_vector(self, subsymbol: str) -> np.ndarray:
        """Get or create vector for a subsymbol."""
        if subsymbol not in self.subsymbol_cache:
            self.subsymbol_cache[subsymbol] = self.factory.get_or_create_vector(f"sub_{subsymbol}")
        return self.subsymbol_cache[subsymbol]
    
    def _circular_convolution(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular convolution for binding operation."""
        return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=len(a))
    
    def compose(self, symbol_tuple: Tuple[str, ...]) -> np.ndarray:
        """Compose a tuple of subsymbols into a single vector."""
        if len(symbol_tuple) == 0:
            raise ValueError("Cannot compose empty tuple")
        
        if len(symbol_tuple) == 1:
            return self._get_subsymbol_vector(symbol_tuple[0])
        
        # Get vectors for all subsymbols
        subsymbol_vecs = [self._get_subsymbol_vector(sub) for sub in symbol_tuple]
        
        if self.method == "addition":
            result = np.sum(subsymbol_vecs, axis=0)
        elif self.method == "multiplication":
            result = subsymbol_vecs[0].copy()
            for vec in subsymbol_vecs[1:]:
                result *= vec
        elif self.method == "binding":
            result = subsymbol_vecs[0].copy()
            for vec in subsymbol_vecs[1:]:
                result = self._circular_convolution(result, vec)
        elif self.method == "position_encoding":
            result = np.zeros(self.factory.vector_dim)
            for i, vec in enumerate(subsymbol_vecs):
                pos_vec = np.zeros(self.factory.vector_dim)
                pos_indices = np.arange(i, self.factory.vector_dim, 
                                      len(subsymbol_vecs))[:self.factory.vector_dim//len(subsymbol_vecs)]
                pos_vec[pos_indices] = 1.0
                pos_vec = pos_vec / np.linalg.norm(pos_vec)
                positioned_vec = self._circular_convolution(vec, pos_vec)
                result += positioned_vec
        else:
            raise ValueError(f"Unknown composition method: {self.method}")
        
        # Normalize result
        return result / np.linalg.norm(result)

class TupleHeteroassociativeMemory:
    """
    Heteroassociative memory that works with tuple symbols.
    """
    
    def __init__(self, 
                 composer: TupleComposer,
                 learning_rate: float = 1.0,
                 retrieval_threshold: float = 0.3,
                 memory_name: str = "memory"):
        """
        Initialize heteroassociative memory for tuples.
        
        Args:
            composer: Tuple composer for vector generation
            learning_rate: Learning rate for associations
            retrieval_threshold: Minimum activation for successful retrieval
            memory_name: Name for debugging/identification
        """
        self.composer = composer
        self.learning_rate = learning_rate
        self.retrieval_threshold = retrieval_threshold
        self.memory_name = memory_name
        
        # Association matrix and learned pairs
        self.association_matrix: Optional[np.ndarray] = None
        self.associations: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
        
        # Cache for composed vectors
        self.tuple_cache: Dict[str, np.ndarray] = {}
    
    def _get_tuple_vector(self, symbol_tuple: Tuple[str, ...]) -> np.ndarray:
        """Get or create vector for a tuple symbol."""
        tuple_key = str(symbol_tuple)
        if tuple_key not in self.tuple_cache:
            self.tuple_cache[tuple_key] = self.composer.compose(symbol_tuple)
        return self.tuple_cache[tuple_key]
    
    def _ensure_matrix_size(self):
        """Ensure association matrix is properly sized."""
        if self.association_matrix is None:
            dim = self.composer.factory.vector_dim
            self.association_matrix = np.zeros((dim, dim))
    
    def learn_association(self, input_tuple: Tuple[str, ...], output_tuple: Tuple[str, ...]):
        """Learn an association between two tuple symbols using Hebbian rule."""
        input_vector = self._get_tuple_vector(input_tuple)
        output_vector = self._get_tuple_vector(output_tuple)
        
        self._ensure_matrix_size()
        
        # Hebbian learning: outer product
        self.association_matrix += self.learning_rate * np.outer(output_vector, input_vector)
        self.associations.append((input_tuple, output_tuple))
    
    def get_activation(self, input_tuple: Tuple[str, ...]) -> np.ndarray:
        """Get the raw activation vector for an input tuple."""
        if self.association_matrix is None:
            return np.zeros(self.composer.factory.vector_dim)
        
        input_vector = self._get_tuple_vector(input_tuple)
        return self.association_matrix @ input_vector

class AssociativeHeadBuilder:
    """
    Builder for constructing associative attention heads via rules.
    Handles the rule-based construction phase.
    """
    
    def __init__(self,
                 vector_dim: int = 512,
                 composition_method: str = "binding",
                 sparsity: float = 1.0,
                 orthogonality_threshold: float = 0.1,
                 learning_rate: float = 1.0,
                 retrieval_threshold: float = 0.3,
                 temperature: float = 1.0):
        """
        Initialize associative head builder.
        
        Args:
            vector_dim: Dimensionality of vectors
            composition_method: How to compose tuples ("binding", "multiplication", etc.)
            sparsity: Vector sparsity (1.0 = dense)
            orthogonality_threshold: Threshold for vector orthogonality
            learning_rate: Learning rate for all memories
            retrieval_threshold: Minimum activation threshold
            temperature: Temperature for attention softmax
        """
        # Shared vector factory
        self.vector_factory = OrthogonalVectorFactory(
            vector_dim=vector_dim,
            sparsity=sparsity,
            orthogonality_threshold=orthogonality_threshold
        )
        
        # Shared tuple composer
        self.composer = TupleComposer(self.vector_factory, composition_method)
        
        # Three heteroassociative memories
        self.key_memory = TupleHeteroassociativeMemory(
            self.composer, learning_rate, retrieval_threshold, "KeyMemory"
        )
        self.query_memory = TupleHeteroassociativeMemory(
            self.composer, learning_rate, retrieval_threshold, "QueryMemory"
        )
        self.value_memory = TupleHeteroassociativeMemory(
            self.composer, learning_rate, retrieval_threshold, "ValueMemory"
        )
        
        # Parameters
        self.temperature = temperature
        self.vector_dim = vector_dim
        
        # Rule tracking
        self.linking_rules: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
        self.value_rules: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
        
        # Symbol library for output interpretation
        self.symbol_library: Dict[str, np.ndarray] = {}
    
    def add_linking_rule(self, query_symbol: Tuple[str, ...], key_symbol: Tuple[str, ...],
                        use_shared_intermediate: bool = True):
        """
        Add a linking rule that creates associations between query and key symbols.
        
        Args:
            query_symbol: Tuple symbol that should attend to key_symbol
            key_symbol: Tuple symbol that should be attended to
            use_shared_intermediate: Whether to use shared intermediate (recommended)
        """
        if use_shared_intermediate:
            # Create shared intermediate symbol
            intermediate = ("link", f"{query_symbol}_{key_symbol}")
            # Learn associations to shared intermediate
            self.query_memory.learn_association(query_symbol, intermediate)
            self.key_memory.learn_association(key_symbol, intermediate)
        else:
            # Create separate intermediates
            query_intermediate = ("query_int", str(query_symbol))
            key_intermediate = ("key_int", str(key_symbol))
            self.query_memory.learn_association(query_symbol, query_intermediate)
            self.key_memory.learn_association(key_symbol, key_intermediate)
        
        self.linking_rules.append((query_symbol, key_symbol))
        print(f"Added linking rule: {query_symbol} <-> {key_symbol}")
    
    def add_value_rule(self, key_symbol: Tuple[str, ...], value_symbol: Tuple[str, ...]):
        """
        Add a value rule that maps key symbols to value symbols.
        
        Args:
            key_symbol: Key symbol that when attended to should retrieve value_symbol
            value_symbol: Value symbol to be retrieved
        """
        self.value_memory.learn_association(key_symbol, value_symbol)
        self.value_rules.append((key_symbol, value_symbol))
        print(f"Added value rule: {key_symbol} -> {value_symbol}")
    
    def add_symbol_to_library(self, symbol: Tuple[str, ...]):
        """Add a symbol to the library for output interpretation."""
        symbol_key = str(symbol)
        if symbol_key not in self.symbol_library:
            self.symbol_library[symbol_key] = self.composer.compose(symbol)
    
    def create_input_tensor(self, symbols: List[Tuple[str, ...]], batch_size: int = 1) -> torch.Tensor:
        """
        Create input tensor from list of symbol tuples.
        
        Args:
            symbols: List of symbol tuples to convert to tensor
            batch_size: Number of batches
            
        Returns:
            Input tensor of shape (batch_size, seq_len, vector_dim)
        """
        seq_len = len(symbols)
        input_tensor = torch.zeros(batch_size, seq_len, self.vector_dim)
        
        for i, symbol in enumerate(symbols):
            # Add to library if not present
            self.add_symbol_to_library(symbol)
            
            # Get composed vector for symbol
            symbol_vector = self.composer.compose(symbol)
            symbol_tensor = torch.from_numpy(symbol_vector).float()
            
            # Set for all batches
            for b in range(batch_size):
                input_tensor[b, i, :] = symbol_tensor
        
        return input_tensor
    
    def build(self) -> 'AssociativeHead':
        """
        Build the final AssociativeHead PyTorch module.
        
        Returns:
            AssociativeHead: Ready-to-use PyTorch attention head
        """
        # Build symbol library from all used symbols
        all_symbols = set()
        
        # Add symbols from linking rules
        for query_sym, key_sym in self.linking_rules:
            all_symbols.add(query_sym)
            all_symbols.add(key_sym)
        
        # Add symbols from value rules  
        for key_sym, value_sym in self.value_rules:
            all_symbols.add(key_sym)
            all_symbols.add(value_sym)
        
        # Add to library
        for symbol in all_symbols:
            self.add_symbol_to_library(symbol)
        
        # Extract matrices
        query_matrix = self.query_memory.association_matrix
        key_matrix = self.key_memory.association_matrix  
        value_matrix = self.value_memory.association_matrix
        
        # Handle None matrices
        if query_matrix is None:
            query_matrix = np.eye(self.vector_dim)
        if key_matrix is None:
            key_matrix = np.eye(self.vector_dim)
        if value_matrix is None:
            value_matrix = np.eye(self.vector_dim)
        
        return AssociativeHead(
            query_matrix=query_matrix,
            key_matrix=key_matrix,
            value_matrix=value_matrix,
            symbol_library=self.symbol_library,
            temperature=self.temperature,
            vector_dim=self.vector_dim,
            builder=self  # Pass builder for testing
        )

class AssociativeHead(nn.Module):
    """
    PyTorch attention head that uses pre-built associative memory matrices.
    Operates on standard (B,T,D) tensors.
    """
    
    def __init__(self,
                 query_matrix: np.ndarray,
                 key_matrix: np.ndarray, 
                 value_matrix: np.ndarray,
                 symbol_library: Dict[str, np.ndarray],
                 temperature: float = 1.0,
                 vector_dim: int = 512,
                 builder: Optional['AssociativeHeadBuilder'] = None):
        """
        Initialize PyTorch attention head with pre-built matrices.
        
        Args:
            query_matrix: Pre-built query transformation matrix
            key_matrix: Pre-built key transformation matrix
            value_matrix: Pre-built value transformation matrix
            symbol_library: Library of known symbols for output interpretation
            temperature: Temperature for attention softmax
            vector_dim: Vector dimensionality
            builder: Reference to builder for testing (optional)
        """
        super().__init__()
        
        # Convert matrices to PyTorch parameters
        self.query_matrix = nn.Parameter(torch.from_numpy(query_matrix).float(), requires_grad=False)
        self.key_matrix = nn.Parameter(torch.from_numpy(key_matrix).float(), requires_grad=False)
        self.value_matrix = nn.Parameter(torch.from_numpy(value_matrix).float(), requires_grad=False)
        
        # Store symbol library
        self.symbol_library = {k: torch.from_numpy(v).float() for k, v in symbol_library.items()}
        self.temperature = temperature
        self.vector_dim = vector_dim
        self.builder = builder  # For testing
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of associative attention.
        
        Args:
            x: Input tensor of shape (B, T, D)
            
        Returns:
            Output tensor of shape (B, T, D)
        """
        B, T, D = x.shape
        
        # Compute queries, keys, values using associative matrices
        # Q = x @ query_matrix^T (since matrix was built as output @ input^T)
        queries = x @ self.query_matrix.T  # (B, T, D)
        keys = x @ self.key_matrix.T       # (B, T, D)
        values = x @ self.value_matrix.T   # (B, T, D)
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.temperature  # (B, T, T)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)  # (B, T, T)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, values)  # (B, T, D)
        
        return output
    
    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns attention weights for analysis.
        
        Args:
            x: Input tensor of shape (B, T, D)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, T, D = x.shape
        
        # Compute queries, keys, values using associative matrices
        queries = x @ self.query_matrix.T  # (B, T, D)
        keys = x @ self.key_matrix.T       # (B, T, D)
        values = x @ self.value_matrix.T   # (B, T, D)
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.temperature  # (B, T, T)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)  # (B, T, T)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, values)  # (B, T, D)
        
        return output, attention_weights
    
    def interpret_output(self, output_vector: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Interpret an output vector by finding closest symbols in the library.
        Assumes output is additive composition of symbols.
        
        Args:
            output_vector: Single output vector of shape (D,)
            top_k: Number of top symbols to return
            
        Returns:
            List of (symbol, similarity_score) tuples
        """
        if output_vector.dim() > 1:
            output_vector = output_vector.squeeze()
        
        similarities = []
        
        for symbol_str, symbol_vec in self.symbol_library.items():
            # Compute cosine similarity
            similarity = F.cosine_similarity(
                output_vector.unsqueeze(0), 
                symbol_vec.unsqueeze(0)
            ).item()
            similarities.append((symbol_str, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def interpret_batch_output(self, output: torch.Tensor, top_k: int = 5) -> List[List[Tuple[str, float]]]:
        """
        Interpret a batch of output vectors.
        
        Args:
            output: Output tensor of shape (B, T, D) or (T, D)
            top_k: Number of top symbols per position
            
        Returns:
            List of interpretations for each position
        """
        if output.dim() == 3:
            # Handle batch dimension - just take first batch item
            output = output[0]  # (T, D)
        
        interpretations = []
        for t in range(output.shape[0]):
            interp = self.interpret_output(output[t], top_k)
            interpretations.append(interp)
        
        return interpretations

def test_associative_rules():
    """
    Test that the associative head correctly implements the rules we defined.
    """
    print("=" * 80)
    print("=== TESTING ASSOCIATIVE RULES ===")
    print("=" * 80)
    
    # Create builder with rules
    builder = AssociativeHeadBuilder(
        vector_dim=256,
        composition_method="binding",
        temperature=1.0
    )
    
    # Define rules
    print("Setting up rules:")
    
    # Linking rules: who should attend to whom
    builder.add_linking_rule(("subject", "person"), ("verb", "action"))
    builder.add_linking_rule(("verb", "see"), ("object", "visible"))
    builder.add_linking_rule(("adjective", "red"), ("object", "ball"))
    
    # Value rules: what should be retrieved when attending
    builder.add_value_rule(("verb", "action"), ("feature", "dynamic"))
    builder.add_value_rule(("object", "visible"), ("feature", "visual"))
    builder.add_value_rule(("object", "ball"), ("feature", "round"))
    
    # Build head
    print("\nBuilding head...")
    head = builder.build()
    
    # Test 1: Simple two-token test
    print("\n" + "="*50)
    print("TEST 1: Two tokens with direct linking")
    print("="*50)
    
    symbols = [("subject", "person"), ("verb", "action")]
    input_tensor = builder.create_input_tensor(symbols)
    
    print(f"Input symbols: {symbols}")
    print(f"Expected: ('subject', 'person') should attend to ('verb', 'action')")
    print(f"Expected output[0] should contain ('feature', 'dynamic')")
    
    output, attention = head.forward_with_attention(input_tensor)
    
    print(f"\nAttention weights:")
    print(f"  {symbols[0]} -> {symbols[0]}: {attention[0,0,0]:.3f}")
    print(f"  {symbols[0]} -> {symbols[1]}: {attention[0,0,1]:.3f}")
    print(f"  {symbols[1]} -> {symbols[0]}: {attention[0,1,0]:.3f}")
    print(f"  {symbols[1]} -> {symbols[1]}: {attention[0,1,1]:.3f}")
    
    interpretations = head.interpret_batch_output(output)
    print(f"\nOutput interpretations:")
    for i, (symbol, interp) in enumerate(zip(symbols, interpretations)):
        print(f"  Position {i} ({symbol}):")
        for sym_str, score in interp[:3]:
            print(f"    {sym_str}: {score:.3f}")
    
    # Test 2: Three-token chain
    print("\n" + "="*50)
    print("TEST 2: Three-token attention chain")
    print("="*50)
    
    symbols = [("verb", "see"), ("object", "visible"), ("adjective", "red")]
    input_tensor = builder.create_input_tensor(symbols)
    
    print(f"Input symbols: {symbols}")
    print(f"Expected: ('verb', 'see') should attend to ('object', 'visible')")
    print(f"Expected output[0] should contain ('feature', 'visual')")
    
    output, attention = head.forward_with_attention(input_tensor)
    
    print(f"\nAttention weights (first row - 'verb see' attending to others):")
    for j, target in enumerate(symbols):
        print(f"  ('verb', 'see') -> {target}: {attention[0,0,j]:.3f}")
    
    interpretations = head.interpret_batch_output(output)
    print(f"\nOutput interpretations:")
    for i, (symbol, interp) in enumerate(zip(symbols, interpretations)):
        print(f"  Position {i} ({symbol}):")
        for sym_str, score in interp[:3]:
            print(f"    {sym_str}: {score:.3f}")
    
    # Test 3: Full sentence test
    print("\n" + "="*50)
    print("TEST 3: Full sentence with multiple rules")
    print("="*50)
    
    symbols = [
        ("subject", "person"), 
        ("verb", "see"), 
        ("adjective", "red"), 
        ("object", "ball")
    ]
    input_tensor = builder.create_input_tensor(symbols)
    
    print(f"Input symbols: {symbols}")
    print("Expected attention patterns:")
    print("  - ('subject', 'person') -> ('verb', 'action') [but 'action' not in input]")
    print("  - ('verb', 'see') -> ('object', 'visible') [but 'visible' not in input]") 
    print("  - ('adjective', 'red') -> ('object', 'ball')")
    
    output, attention = head.forward_with_attention(input_tensor)
    
    print(f"\nAttention matrix:")
    print("     " + "".join(f"{str(s)[:12]:>12}" for s in symbols))
    for i, query in enumerate(symbols):
        row_str = f"{str(query)[:12]:>12} "
        row_str += "".join(f"{attention[0,i,j]:>12.3f}" for j in range(len(symbols)))
        print(row_str)
    
    interpretations = head.interpret_batch_output(output)
    print(f"\nOutput interpretations:")
    for i, (symbol, interp) in enumerate(zip(symbols, interpretations)):
        print(f"  Position {i} ({symbol}):")
        for sym_str, score in interp[:3]:
            print(f"    {sym_str}: {score:.3f}")
    
    print("\n" + "="*80)
    print("Analysis:")
    print("- High attention weights indicate successful rule matching")
    print("- Output interpretations show retrieved features/values")
    print("- Position 2 ('adjective', 'red') should attend strongly to position 3 ('object', 'ball')")
    print("- Position 3's output should then contain ('feature', 'round')")
    print("="*80)

# Example usage and testing
if __name__ == "__main__":
    print("=== Associative Attention Head Builder Demo ===\n")
    
    # Phase 1: Build the head with rules
    print("Phase 1: Building head with rules...")
    builder = AssociativeHeadBuilder(
        vector_dim=256,
        composition_method="binding",
        temperature=1.0
    )
    
    # Add linking rules (what should attend to what)
    print("\nAdding linking rules:")
    builder.add_linking_rule(("subject", "person"), ("verb", "action"))
    builder.add_linking_rule(("verb", "see"), ("object", "visible"))
    builder.add_linking_rule(("adjective", "color"), ("object", "thing"))
    
    # Add value rules (what should be retrieved when attending)
    print("\nAdding value rules:")
    builder.add_value_rule(("verb", "action"), ("feature", "dynamic"))
    builder.add_value_rule(("object", "visible"), ("feature", "visual"))
    builder.add_value_rule(("object", "thing"), ("feature", "physical"))
    
    # Build the PyTorch head
    print("\nBuilding PyTorch module...")
    head = builder.build()
    
    # Phase 2: Use as standard PyTorch attention head
    print("\nPhase 2: Using as PyTorch attention head...")
    
    # Create input tensors
    batch_size, seq_len, vector_dim = 2, 4, 256
    x = torch.randn(batch_size, seq_len, vector_dim)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = head(x)
    print(f"Output shape: {output.shape}")
    
    # Interpret outputs
    print("\nInterpreting outputs (first batch, all positions):")
    interpretations = head.interpret_batch_output(output)
    
    for i, interp in enumerate(interpretations):
        print(f"Position {i}:")
        for symbol, score in interp:
            print(f"  {symbol}: {score:.3f}")
        print()
    
    # Run comprehensive rule testing
    test_associative_rules() 