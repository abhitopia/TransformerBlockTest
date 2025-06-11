#!/usr/bin/env python3
"""
Associative Attention Head

An attention head implementation that uses heteroassociative memories as the
underlying substrate for Key, Query, and Value operations. Symbols are defined
as tuples of subsymbols, and the head is constructed via rules.

Architecture:
- Key Memory: Maps input symbols to key representations
- Query Memory: Maps input symbols to query representations  
- Value Memory: Maps key symbols to value representations
- Attention computed via associative strength between queries and keys

Example usage:
    head = AssociativeHead(vector_dim=512, composition_method="binding")
    
    # Add linking rules (affects Key and Query memories)
    head.add_linking_rule(("subject", "person"), ("object", "person"))
    head.add_linking_rule(("verb", "see"), ("object", "visible"))
    
    # Add value rules (affects Value memory)
    head.add_value_rule(("object", "person"), ("feature", "animate"))
    head.add_value_rule(("object", "visible"), ("feature", "visual"))
    
    # Forward pass
    tokens = [("subject", "person"), ("verb", "see"), ("object", "book")]
    output = head.forward(tokens)
"""

import numpy as np
import torch.nn as nn
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
    
    def recall(self, input_tuple: Tuple[str, ...]) -> Optional[Tuple[str, ...]]:
        """Recall the tuple associated with the input tuple."""
        if self.association_matrix is None:
            return None
        
        input_vector = self._get_tuple_vector(input_tuple)
        activation = self.association_matrix @ input_vector
        
        # Find best match among learned output tuples
        best_match = None
        best_similarity = self.retrieval_threshold
        
        for _, output_tuple in self.associations:
            output_vector = self._get_tuple_vector(output_tuple)
            similarity = np.dot(activation, output_vector)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = output_tuple
        
        return best_match
    
    def get_activation(self, input_tuple: Tuple[str, ...]) -> np.ndarray:
        """Get the raw activation vector for an input tuple."""
        if self.association_matrix is None:
            return np.zeros(self.composer.factory.vector_dim)
        
        input_vector = self._get_tuple_vector(input_tuple)
        return self.association_matrix @ input_vector
    
    def get_association_strength(self, input_tuple: Tuple[str, ...], 
                                output_tuple: Tuple[str, ...]) -> float:
        """Get the strength of association between two tuples."""
        if self.association_matrix is None:
            return 0.0
        
        input_vector = self._get_tuple_vector(input_tuple)
        output_vector = self._get_tuple_vector(output_tuple)
        
        activation = self.association_matrix @ input_vector
        return np.dot(activation, output_vector)

class AssociativeHead(nn.Module):
    """
    Attention head implemented using heteroassociative memories.
    
    The head consists of three heteroassociative memories:
    - Key Memory: Maps input symbols to key representations
    - Query Memory: Maps input symbols to query representations  
    - Value Memory: Maps key symbols to value representations
    
    Attention is computed via associative strength between queries and keys.
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
        Initialize associative attention head.
        
        Args:
            vector_dim: Dimensionality of vectors
            composition_method: How to compose tuples ("binding", "multiplication", etc.)
            sparsity: Vector sparsity (1.0 = dense)
            orthogonality_threshold: Threshold for vector orthogonality
            learning_rate: Learning rate for all memories
            retrieval_threshold: Minimum activation threshold
            temperature: Temperature for attention softmax
        """
        super().__init__()
        
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
        
        # Attention parameters
        self.temperature = temperature
        self.vector_dim = vector_dim
        
        # Rule tracking
        self.linking_rules: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
        self.value_rules: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
    
    def add_linking_rule(self, query_symbol: Tuple[str, ...], key_symbol: Tuple[str, ...],
                        use_shared_intermediate: bool = True):
        """
        Add a linking rule that creates associations between query and key symbols.
        
        This implements the dual-memory linkage strategy where both query and key
        symbols are linked to the same intermediate representation for strong correlation.
        
        Args:
            query_symbol: Tuple symbol that should attend to key_symbol
            key_symbol: Tuple symbol that should be attended to
            use_shared_intermediate: Whether to use shared intermediate (recommended)
        """
        if use_shared_intermediate:
            # Create shared intermediate symbol
            intermediate = ("link", f"{query_symbol}_{key_symbol}")
        else:
            # Create separate intermediates
            query_intermediate = ("query_int", str(query_symbol))
            key_intermediate = ("key_int", str(key_symbol))
            
            self.query_memory.learn_association(query_symbol, query_intermediate)
            self.key_memory.learn_association(key_symbol, key_intermediate)
            self.linking_rules.append((query_symbol, key_symbol))
            return
        
        # Learn associations to shared intermediate
        self.query_memory.learn_association(query_symbol, intermediate)
        self.key_memory.learn_association(key_symbol, intermediate)
        
        self.linking_rules.append((query_symbol, key_symbol))
        
        print(f"Added linking rule: {query_symbol} <-> {key_symbol} via {intermediate}")
    
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
    
    def compute_attention_weights(self, tokens: List[Tuple[str, ...]]) -> np.ndarray:
        """
        Compute attention weights between all pairs of tokens using associative strength.
        
        Args:
            tokens: List of tuple symbols (input sequence)
            
        Returns:
            Attention weight matrix of shape (seq_len, seq_len)
        """
        seq_len = len(tokens)
        attention_matrix = np.zeros((seq_len, seq_len))
        
        for i, query_token in enumerate(tokens):
            # Get query activation
            query_activation = self.query_memory.get_activation(query_token)
            
            for j, key_token in enumerate(tokens):
                # Get key activation  
                key_activation = self.key_memory.get_activation(key_token)
                
                # Compute attention as dot product of activations
                attention_score = np.dot(query_activation, key_activation)
                attention_matrix[i, j] = attention_score
        
        # Apply temperature and softmax
        attention_matrix = attention_matrix / self.temperature
        
        # Softmax per row
        for i in range(seq_len):
            row = attention_matrix[i, :]
            row_max = np.max(row)
            exp_row = np.exp(row - row_max)  # Numerical stability
            attention_matrix[i, :] = exp_row / np.sum(exp_row)
        
        return attention_matrix
    
    def forward(self, tokens: List[Tuple[str, ...]]) -> List[np.ndarray]:
        """
        Forward pass of the associative attention head.
        
        Args:
            tokens: List of tuple symbols (input sequence)
            
        Returns:
            List of attended output vectors, one per input token
        """
        # Compute attention weights
        attention_weights = self.compute_attention_weights(tokens)
        
        # Get value vectors for all tokens
        value_vectors = []
        for token in tokens:
            # Try to get value via value memory
            value_activation = self.value_memory.get_activation(token)
            
            # If no strong value activation, use the token's own vector
            if np.linalg.norm(value_activation) < 0.1:
                value_activation = self.composer.compose(token)
            
            value_vectors.append(value_activation)
        
        # Compute attended outputs
        attended_outputs = []
        for i in range(len(tokens)):
            attended_vector = np.zeros(self.vector_dim)
            
            # Weighted sum of values
            for j in range(len(tokens)):
                attended_vector += attention_weights[i, j] * value_vectors[j]
            
            attended_outputs.append(attended_vector)
        
        return attended_outputs
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the attention head."""
        return {
            'vector_dim': self.vector_dim,
            'composition_method': self.composer.method,
            'num_linking_rules': len(self.linking_rules),
            'num_value_rules': len(self.value_rules),
            'num_symbols': len(self.vector_factory.vectors),
            'key_memory_associations': len(self.key_memory.associations),
            'query_memory_associations': len(self.query_memory.associations),
            'value_memory_associations': len(self.value_memory.associations),
            'linking_rules': self.linking_rules,
            'value_rules': self.value_rules
        }
    
    def visualize_attention(self, tokens: List[Tuple[str, ...]], 
                           attention_weights: Optional[np.ndarray] = None) -> None:
        """Visualize attention weights as a matrix."""
        if attention_weights is None:
            attention_weights = self.compute_attention_weights(tokens)
        
        print("\nAttention Matrix:")
        print("Rows = Query tokens, Columns = Key tokens")
        print("    " + "  ".join(f"{str(t)[:8]:>8}" for t in tokens))
        
        for i, query_token in enumerate(tokens):
            row_str = f"{str(query_token)[:8]:>8} "
            row_str += "  ".join(f"{attention_weights[i,j]:>8.3f}" for j in range(len(tokens)))
            print(row_str)

# Example usage and testing
if __name__ == "__main__":
    print("=== Associative Attention Head Demo ===\n")
    
    # Create attention head
    head = AssociativeHead(
        vector_dim=256,
        composition_method="binding",
        temperature=1.0
    )
    
    print("Creating attention head with associative memories...")
    
    # Add some linking rules (what should attend to what)
    print("\nAdding linking rules:")
    head.add_linking_rule(("subject", "person"), ("verb", "action"))
    head.add_linking_rule(("verb", "see"), ("object", "visible"))
    head.add_linking_rule(("adjective", "color"), ("object", "thing"))
    
    # Add value rules (what should be retrieved when attending)
    print("\nAdding value rules:")
    head.add_value_rule(("verb", "action"), ("feature", "dynamic"))
    head.add_value_rule(("object", "visible"), ("feature", "visual"))
    head.add_value_rule(("object", "thing"), ("feature", "physical"))
    
    # Test with a sentence
    print("\nTesting attention on sentence tokens:")
    tokens = [
        ("subject", "person"),
        ("verb", "see"),
        ("adjective", "red"),
        ("object", "ball")
    ]
    
    print(f"Input tokens: {tokens}")
    
    # Compute attention
    attention_weights = head.compute_attention_weights(tokens)
    head.visualize_attention(tokens, attention_weights)
    
    # Forward pass
    outputs = head.forward(tokens)
    print(f"\nOutput vectors shape: {[out.shape for out in outputs]}")
    
    # Show statistics
    stats = head.get_statistics()
    print(f"\nHead Statistics:")
    print(f"  Vector dimension: {stats['vector_dim']}")
    print(f"  Composition method: {stats['composition_method']}")
    print(f"  Linking rules: {stats['num_linking_rules']}")
    print(f"  Value rules: {stats['num_value_rules']}")
    print(f"  Total symbols: {stats['num_symbols']}") 