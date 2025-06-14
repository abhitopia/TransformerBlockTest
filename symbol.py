from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Union
import numpy as np
import torch


BASE_LABEL = str
COMPOSITE_LABEL = Tuple[BASE_LABEL, ...]
CONCATENATED_LABEL = Tuple[Union[BASE_LABEL, COMPOSITE_LABEL], ...]
LABEL = Union[BASE_LABEL, COMPOSITE_LABEL, CONCATENATED_LABEL]



class SymbolType(Enum):
    BASE = "base"
    COMPOSITE = "composite"
    CONCATENATED = "concatenated"

@dataclass
class Symbol:
    label: LABEL
    vec: np.ndarray

    @property
    def dim(self) -> int:
        return self.vec.shape[0]

    @property
    def stype(self) -> SymbolType:
        if isinstance(self.label, BASE_LABEL):
            return SymbolType.BASE
        elif isinstance(self.label, COMPOSITE_LABEL):
            return SymbolType.COMPOSITE
        elif isinstance(self.label, CONCATENATED_LABEL):
            return SymbolType.CONCATENATED
        else:
            raise ValueError(f"Undefined label type: {type(self.label)}")


class OrthogonalVectorFactory:
    def __init__(self, dim: int, sparsity: float = 1.0, threshold: float = 0.1, max_attempts: int = 100, normalize: bool = True):
        self.dim = dim
        self.sparsity = sparsity
        self.threshold = threshold
        self.max_attempts = max_attempts
        self.normalize = normalize
        self.vectors: np.ndarray = np.zeros((0, dim))
        self.max_similarity = 0.0

    def _generate_vector(self):
        vector = np.zeros(self.dim)
        n_nonzero = max(1, int(self.dim * self.sparsity))
        # Random positions for non-zero elements
        positions = np.random.choice(self.dim, n_nonzero, replace=False)
        values = np.random.randn(n_nonzero)
        vector[positions] = values

        if self.normalize:
            vector = vector / np.linalg.norm(vector)

        vector_matrix = np.vstack((self.vectors, vector))
        similarities = np.abs(vector_matrix @ vector)
        max_similarity = np.max(similarities)

        return vector, max_similarity

    def create(self):
        for _ in range(self.max_attempts):
            vector, max_similarity = self._generate_vector()
            if max_similarity < self.threshold:
                break
        self.max_similarity = max(self.max_similarity, max_similarity)
        self.vectors = np.vstack((self.vectors, vector))
        return vector
    
    def add_vector(self, vector: np.ndarray):
        self.vectors = np.vstack((self.vectors, vector))
        self.max_similarity = max(self.max_similarity, np.max(np.abs(self.vectors @ vector)))
        return self.max_similarity
    

class SymbolComposer:
    def __init__(self, strategy: str = "binding"):
        self.strategy = strategy

    def _circular_convolution(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular convolution for binding operation."""
        return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=len(a))

    def __call__(self, symbols: Tuple[Symbol]) -> Symbol:
        assert all(symbol.stype == SymbolType.BASE for symbol in symbols), "All symbols must be base labels"
        assert all(symbol.dim == symbols[0].dim for symbol in symbols), "All symbols must have the same dimension"

        dim = symbols[0].dim


        symbol_vecs = [symbol.vec for symbol in symbols]
        if self.method == "addition":
            result = np.sum(symbol_vecs, axis=0)
        elif self.method == "multiplication":
            result = symbol_vecs[0].copy()
            for vec in symbol_vecs[1:]:
                result *= vec
        elif self.method == "binding":
            result = symbol_vecs[0].copy()
            for vec in symbol_vecs[1:]:
                result = self._circular_convolution(result, vec)
        elif self.method == "position_encoding":
            result = np.zeros(dim)
            for i, vec in enumerate(symbol_vecs):
                pos_vec = np.zeros(dim)
                pos_indices = np.arange(i, dim, 
                                      len(symbol_vecs))[:self.factory.vector_dim//len(symbol_vecs)]
                pos_vec[pos_indices] = 1.0
                pos_vec = pos_vec / np.linalg.norm(pos_vec)
                positioned_vec = self._circular_convolution(vec, pos_vec)
                result += positioned_vec
        else:
            raise ValueError(f"Unknown composition method: {self.method}")

        return Symbol(label=tuple(s.label for s in symbols), vec=result)

class SymbolRegistry:
    def __init__(self, dim: int, sparsity: float = 1.0, threshold: float = 0.1, max_attempts: int = 100, normalize: bool = True, strategy: str = "binding"):
        self.dim = dim
        self.symbols: Dict[LABEL, Symbol] = {}
        self.vector_factory = OrthogonalVectorFactory(dim=dim, sparsity=sparsity, threshold=threshold, max_attempts=max_attempts, normalize=normalize)
        self.composer = SymbolComposer(strategy=strategy)

    def __getitem__(self, label: LABEL) -> Symbol:
        return self.symbols[label]
    
    def register(self, label: LABEL) -> Symbol:
        assert label not in self.symbols, f"Symbol {label} already exists"
        if isinstance(label, BASE_LABEL):
            orth_vec = self.vector_factory.create()
            self.symbols[label] = Symbol(label, orth_vec)
        elif isinstance(label, COMPOSITE_LABEL):
            base_symbols = [self[base_label] for base_label in label]
            composed_symbol = self.composer(base_symbols)
            self.symbols[label] = composed_symbol
        else:
            raise ValueError(f"Undefined label type: {type(label)}")

        return self.symbols[label]
    
    def __setitem__(self, label: LABEL, symbol: Symbol):
        self.symbols[label] = symbol
        self.vector_factory.add_vector(symbol.vec)
    
        
class BlockModule:
    def __call__(self, input_symbols: SymbolRegistry) -> Tuple[SymbolRegistry]:
        """
        Returns a tuple of (output_symbols, and constructed module)
        """
        pass


class SymbolSplit(BlockModule):
    def __init__(self, num_splits: int):
        self.num_splits = num_splits
    
    def __call__(self, input_symbols: SymbolRegistry) -> Tuple[SymbolRegistry]:
        assert input_symbols.dim % self.num_splits == 0, "dim must be divisible by num_splits"
        d_head = input_symbols.dim // self.num_splits
        output_registries = [SymbolRegistry(dim=d_head) for _ in range(self.num_splits)]
        for label, symbol in input_symbols.symbols.items():
            # split the vector into num_splits parts
            parts = np.split(symbol.vec, self.num_splits)
            for i, part in enumerate(parts):
                output_registries[i][label] = Symbol(label, part)
      
        return output_registries



class HeterAssociativeMemory(BlockModule):
    def __init__(self, dim: int, learning_rate: float = 1.0):
        self.dim = dim
        self.memory = np.zeros((dim, dim))
        self.lr = learning_rate
    
    def __call__(self, input_symbols: SymbolRegistry, associations: List[Tuple[LABEL, LABEL]]) -> Tuple[SymbolRegistry]:
        output_symbols = SymbolRegistry(dim=self.dim) # TODO: Allow a global setting for the registry

        for input_label, output_label in associations:
            input_symbol = input_symbols[input_label]
            output_symbol = output_symbols.register(output_label)
            self.memory += self.lr * np.outer(input_symbol.vec, output_symbol.vec)
        return output_symbols


@dataclass
class Link:
    query_label: LABEL
    key_label: LABEL

@dataclass
class Value:
    label: LABEL
    value_label: LABEL

class LinkedDualHeteroAssociativeMemory(BlockModule):
    def __init__(self, dim: int, learning_rate: float = 1.0):
        self.dim = dim
        self.W_q = np.zeros((dim, dim))
        self.W_k = np.zeros((dim, dim))
        self.lr = learning_rate
    
    def __call__(self, input_symbols: SymbolRegistry, links: List[Link]) -> Tuple[SymbolRegistry]:
        output_symbols = SymbolRegistry(dim=self.dim)

        for link in links:
            query_symbol = input_symbols[link.query_label]
            key_symbol = input_symbols[link.key_label]

            interim_label = f"{link.query_label}_{link.key_label}"
            interim_symbol = output_symbols.register(interim_label)
            self.W_q += self.lr * np.outer(query_symbol.vec, interim_symbol.vec)
            self.W_k += self.lr * np.outer(key_symbol.vec, interim_symbol.vec)

        return output_symbols

class SymbolHead(BlockModule):
    def __init__(self, d_head: int):
        self.d_head = d_head
        self.W_qk = LinkedDualHeteroAssociativeMemory(dim=d_head)
        self.W_v = HeterAssociativeMemory(dim=d_head)
    
    def __call__(self, input_symbols: SymbolRegistry, links: List[Link], values: List[Value]) -> Tuple[SymbolRegistry]:
        value_symbols = self.W_v(input_symbols, values)
        interm_symbols = self.W_qk(input_symbols, links)
        return value_symbols

class SymbolConcatenate(BlockModule):
    pass

class SymbolResidual(BlockModule):
    pass

class SymbolTransformerBlock(BlockModule):
    def __init__(self, n_heads: int, d_model: int):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        d_head = d_model // n_heads
        self.heads = [SymbolHead(d_head) for _ in range(n_heads)]
        self.split = SymbolSplit(n_heads)
        self.concat = SymbolConcatenate()
        self.residual = SymbolResidual()
        pass
    

    def __call__(self, input_symbols: SymbolRegistry, head_links: Tuple[List[Link]], head_values: Tuple[List[Value]]) -> Tuple[SymbolRegistry]:
        head_inputs = self.split(input_symbols)
        assert len(head_links) == len(head_values) == len(self.heads), "head_links, head_values and heads must have the same length"
        head_outputs = []
        for head_input, head, head_link, head_value in zip(head_inputs, self.heads, head_links, head_values):
            head_output = head(head_input, links=head_link, values=head_value)
            head_outputs.append(head_output)
        
        # combined_output = self.concat(head_outputs)
        # residual = self.residual(input_symbols, combined_output)


def create_multiplier(M: int, N: int, T: int, n_heads: int, d_model: int):
    assert T >= M+N, "number of token positions must be greater than or equal to the sum of multiplier and multiplicand positions"

    # Create base symbols
    symbols = SymbolRegistry(dim=d_model)
    for d in range(10):
        symbols.register(f"D{d}")

    # Position of the digit
    for i in range(M+N):
        symbols.register(f"P{i}")
    symbols.register("Multiplicand")
    symbols.register("Multiplier")

    # Token positions
    for i in range(T):
        symbols.register(f"T{i}")

    m = max(M, N)
    n = min(M, N)


    # Example
    #      1234
    #    x  567
    # ---------  L0
    # H0:  7777
    # H1:  1234
    # H2:  0000  (Previous Carry)
    # ---------
    # FF:  7418  (Base Multiply)
    #      0122  (Carry)
    # --------- L1
    # 

    # number of layers is n
    for layer in range(1):
        transformer_block = SymbolTransformerBlock(n_heads=n_heads, d_model=d_model)
        multiplier_pos = f"P{layer}"

        # Head 0: Collect active multiplier digit for each token position
        head_0_links = []
        for tloc in range(T):
            # For each of first m positions, we want to collect the active multiplier digit
            if tloc < m:
                query_label = f"T{tloc}"
                key_label = ("Multiplier", multiplier_pos)
                head_0_links.append(Link(query_label=query_label, key_label=key_label))

        head_0_values = []
        # collect the digit from multiplier and multiplicand
        for d in range(10):
            head_0_values.append(Value(label=("Multiplier", f"D{d}"), value_label=f"Multiplier{d}"))

        # Head 1: Collect all the multiplicand digits
        head_1_links = []
        for tloc in range(T):
            if tloc < m:
                query_label = f"T{tloc}"
                key_label = ("Multiplicand", f"P{tloc}") 
                head_1_links.append(Link(query_label=query_label, key_label=key_label))

        for d in range(10):
            head_0_values.append(Value(label=("Multiplicand", f"D{d}"), value_label=f"Multiplicand{d}"))


        # Head 2: Matches with previous token and extract the carry digit
        head_2_links = []
        for tloc in range(T):
            if tloc > 0:
                query_label = f"T{tloc}"
                key_label = f"T{tloc-1}"
                head_2_links.append(Link(query_label=query_label, key_label=key_label))

        head_2_values = []
        # For first layer, carry is always 0
        if layer == 0:
            for tloc in range(T):
                head_2_values.append(Value(label=f"T{tloc}", value_label=f"Carry{0}"))
        else:
            for d in range(10):
                head_2_values.append(Value(label=f"Carry{d}", value_label=f"Carry{d}"))

        

if __name__ == "__main__":

    create_multiplier(M=3, N=2, n_heads=4, d_model=128)
    pass