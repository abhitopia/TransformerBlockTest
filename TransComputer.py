import math
from torch import nn
import torch
from torch.nn import functional as F
from dataclasses import dataclass


class GatedCrossAttention(nn.Module):
    """
    PreNorm version of the Gated Cross-Attention block (Flamigo Style).
    Applies LayerNorm before each sublayer (cross-attention and feed-forward),
    following a typical "pre-norm" Transformer design.

    Args:
        embed_dim (int): Dimension of input embeddings (d_model).
        num_heads (int): Number of attention heads.
        ff_hidden_dim (int): Hidden dimension for the feed-forward network.
        dropout (float): Dropout probability (optional).
    """
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.0):
        super(GatedCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # ------------------------------------------------------------
        # 1) PreNorm LayerNorm for Cross-Attention
        # ------------------------------------------------------------
        self.ln1 = nn.LayerNorm(embed_dim)

        # Cross-Attention module
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # Gate (scalar) for cross-attn, initialized to small positive value to encourage usage
        self.gate_xattn = nn.Parameter(torch.ones(1) * 0.1)

        # ------------------------------------------------------------
        # 2) PreNorm LayerNorm for Feed-Forward
        # ------------------------------------------------------------
        self.ln2 = nn.LayerNorm(embed_dim)

        # Feed-Forward network (two-layer MLP)
        self.fc1 = nn.Linear(embed_dim, ff_hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(ff_hidden_dim, embed_dim)
        # Gate (scalar) for dense sublayer, initialized to small positive value to encourage usage  
        self.gate_dense = nn.Parameter(torch.ones(1) * 0.1)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, hidden_states, context_tokens, attn_mask=None, key_padding_mask=None):
        """
        Args:
            hidden_states (Tensor): shape (batch, seq_len, embed_dim). 
                Input hidden states (pre-LM cross-attn) – not assumed to be normalized.
            context_tokens (Tensor): shape (batch, context_len, embed_dim).
            attn_mask (Tensor, optional): shape (seq_len, context_len).
            key_padding_mask (Tensor, optional): shape (batch, context_len).

        Returns:
            Tensor: same shape as hidden_states, after prenorm gated cross-attn + prenorm gated FFN.
        """

        # ======== SUBLAYER 1: Gated Cross-Attention =========
        # PreNorm
        x_norm1 = self.ln1(hidden_states)

        # Cross-Attention: query = x_norm1, key/value = visual_tokens
        attn_output, _ = self.cross_attn(
            query=x_norm1, 
            key=context_tokens, 
            value=context_tokens, 
            attn_mask=attn_mask, 
            key_padding_mask=key_padding_mask
        )
        # Gate the cross-attn output
        gate_val_xattn = torch.tanh(self.gate_xattn)
        gated_attn = gate_val_xattn * attn_output

        # Residual connection
        x = hidden_states + gated_attn  # Output of sublayer1

        # ======== SUBLAYER 2: Gated Feed-Forward (Dense) =========
        # PreNorm
        x_norm2 = self.ln2(x)

        # Two-layer MLP
        ff_intermediate = self.activation(self.fc1(x_norm2))
        ff_output = self.fc2(ff_intermediate)
        if self.dropout is not None:
            ff_output = self.dropout(ff_output)

        # Gate the FFN output
        gate_val_dense = torch.tanh(self.gate_dense)
        gated_ff = gate_val_dense * ff_output

        # Residual connection
        out = x + gated_ff  # Final output

        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, d_model, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout: float = 0.0, is_causal: bool = False):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.is_causal = is_causal
        self.dropout_p = dropout

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3 * d_model)
        qkv = qkv.view(batch_size, seq_len, self.num_heads, 3 * self.head_dim)  # (B, T, n_head, 3 * head_dim)
        qkv = qkv.transpose(1, 2)  # (B, n_head, T, 3*head_dim)
        q, k, v = qkv.split(self.head_dim, dim=-1)  # (B, n_head, T, head_dim)

        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p, is_causal=self.is_causal)  # (B, n_head, T, head_dim)

        # reorder back to (B, T, n_head, head_dim) then merge heads → (B, T, d_model)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (B, T, d_model)
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # final linear + dropout
        out = self.out_proj(attn_output)
        out = F.dropout(out, p=self.dropout_p)
        return out
    

class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, d_model, n_head, dropout: float = 0.0, is_causal: bool = False):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadSelfAttention(d_model, n_head, dropout=dropout, is_causal=is_causal)
        self.ffwd = FeedFoward(d_model, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class ComputeBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout: float = 0.0, is_causal: bool = False, num_symbols: int = 100, include_ffn: bool = False, causal_cross_head: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = n_head
        self.head_dim = d_model // n_head
        self.num_symbols = num_symbols
        self.include_ffn = include_ffn
        self.causal_cross_head = causal_cross_head
        
        # Symbol embeddings only if using symbol-based approach
        if num_symbols > 0:
            self.symbol_embeddings = nn.Parameter(torch.randn(num_symbols, self.head_dim).unsqueeze(0).unsqueeze(-2) * 0.02)
            self.kv_proj = nn.Linear(self.head_dim, 2 * self.head_dim, bias=False)
        else:
            # Input-derived approach: project input to keys and values
            self.symbol_embeddings = None
            self.kv_proj = nn.Linear(self.head_dim, 2 * self.head_dim, bias=False)
        
        self.ln1 = nn.LayerNorm(self.head_dim)

        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.q_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.is_causal = is_causal
        self.dropout_p = dropout
        self.qkv_proj_h = nn.Linear(self.head_dim, 3 * self.head_dim, bias=False)

        if self.include_ffn:
            self.ln2 = nn.LayerNorm(self.head_dim)
            self.ffn = FeedFoward(self.head_dim, dropout=dropout)


    def forward(self, x, K: int):
        batch_size, seq_len, _ = x.size()

        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)  # (B, T, n_head, head_dim)
        q = self.q_proj(self.ln1(x))  # (B, T, n_head, head_dim)
        
        if self.num_symbols > 0:
            # Symbol-based approach (original)
            k, v = self.kv_proj(self.symbol_embeddings).expand(batch_size, -1, self.num_heads, -1).split(self.head_dim, dim=-1)  # (B, num_symbols, n_head, head_dim)
        else:
            # Input-derived approach: keys and values from input
            kv_input = self.ln1(x)  # Normalize input for K,V projection
            k, v = self.kv_proj(kv_input).split(self.head_dim, dim=-1)  # (B, T, n_head, head_dim)

        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)  # (B, n_head, *, head_dim) - * is num_symbols or T
        v = v.transpose(1, 2)  # (B, n_head, *, head_dim) - * is num_symbols or T

        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p, is_causal=self.is_causal)  # (B, n_head, T, head_dim)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (B, T, n_head, head_dim) 

        # Now do attention across heads
        qkv_h = self.qkv_proj_h(attn_output)  # (B, T, n_head, 3 * head_dim)
        q_h, k_h, v_h = qkv_h.split(self.head_dim, dim=-1)  # (B, T, n_head, head_dim)

        attn_output_h = F.scaled_dot_product_attention(q_h, k_h, v_h, dropout_p=self.dropout_p, is_causal=self.causal_cross_head)  # (B, T, n_head, head_dim)
        x = x + attn_output_h
        if self.include_ffn:
            x = x + self.ffn(self.ln2(x))

        x = x.view(batch_size, seq_len, self.d_model)
        return x


# class ComputeBlock(nn.Module):
#     """
#     A ComputeBlock that, when unrolled K times, applies each residual with uniform 1/K scaling.
    
#     1) First attention (“symbol library”): each head attends over num_symbols.
#     2) Second attention (“across heads”): for each token, its n_head head-vectors attend to each other.
#     3) Optional FFN on each head-slice.
    
#     Usage:
#         compute_block = UniformScaledSymbolicComputeBlock(...)
#         p = prev_p  # shape (B, T, d_model)
#         for _ in range(K):
#             p = compute_block(p, K)
#         next_p_i = p
#     """
#     def __init__(
#         self,
#         d_model: int,
#         n_head: int,
#         num_symbols: int,
#         include_ffn: bool = False,
#         dropout: float = 0.0,
#         is_causal: bool = False
#     ):
#         super().__init__()
#         assert d_model % n_head == 0, "d_model must be divisible by n_head"
#         self.d_model      = d_model
#         self.num_heads    = n_head
#         self.head_dim     = d_model // n_head
#         self.num_symbols  = num_symbols
#         self.include_ffn  = include_ffn
#         self.is_causal    = is_causal
#         self.dropout_p    = dropout

#         # Learned symbol-library embeddings: shape (1, num_symbols, 1, head_dim)
#         self.symbol_embeddings = nn.Parameter(
#             torch.randn(num_symbols, self.head_dim)
#             .unsqueeze(0).unsqueeze(-2) * 0.02
#         )

#         # LayerNorm before first attention (per head-dimension)
#         self.ln1 = nn.LayerNorm(self.head_dim)
#         # Linear projections for Q (head_dim→head_dim) and K/V (head_dim→2*head_dim)
#         self.q_proj  = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.kv_proj = nn.Linear(self.head_dim, 2 * self.head_dim, bias=False)

#         # For second attention (“across heads”), project each head-slice to Q/K/V
#         self.qkv_proj_h = nn.Linear(self.head_dim, 3 * self.head_dim, bias=False)

#         # Learned gates for uniform 1/K scaling (initialized to 0 so tanh(0)=0)
#         self.alpha_attn  = nn.Parameter(torch.ones(1)*math.atanh(0.01))  # gate for first attention
#         self.alpha_attn2 = nn.Parameter(torch.ones(1)*math.atanh(0.01))  # gate for second attention
#         if self.include_ffn:
#             self.alpha_ffn = nn.Parameter(torch.ones(1)*math.atanh(0.01))  # gate for FFN
#             self.ln2       = nn.LayerNorm(self.head_dim)
#             self.ffn       = FeedFoward(self.head_dim, dropout=dropout)

#     def forward(self, x: torch.Tensor, K: int) -> torch.Tensor:
#         """
#         Args:
#             x: Tensor of shape (B, T, d_model)
#             K: int, number of compute steps to unroll
#         Returns:
#             Tensor of shape (B, T, d_model) after one iteration;
#             caller should loop K times.
#         """
#         B, T, D = x.size()
#         device  = x.device

#         # Compute uniform 1/K scaling factors
#         scale1 = torch.tanh(self.alpha_attn)  / float(K)  # for first attention
#         scale2 = torch.tanh(self.alpha_attn2) / float(K)  # for second attention
#         if self.include_ffn:
#             scale3 = torch.tanh(self.alpha_ffn) / float(K)  # for FFN

#         # === 1) First attention: each head attends over the symbol library ===
#         # Reshape x → (B, T, n_head, head_dim)
#         x_heads = x.view(B, T, self.num_heads, self.head_dim)

#         # Prepare Q from x: apply LayerNorm then linear
#         q = self.q_proj(self.ln1(x_heads))  # shape (B, T, n_head, head_dim)

#         # Prepare K, V from symbol_embeddings: (1, num_symbols, 2*head_dim)
#         kv = self.kv_proj(self.symbol_embeddings)
#         # Expand → (B, num_symbols, n_head, 2*head_dim)
#         kv = kv.expand(B, -1, self.num_heads, -1)
#         k, v = kv.split(self.head_dim, dim=-1)  # each: (B, num_symbols, n_head, head_dim)

#         # Transpose to (B, n_head, T, head_dim) and (B, n_head, num_symbols, head_dim)
#         q_a = q.transpose(1, 2)  # (B, n_head, T, head_dim)
#         k_a = k.transpose(1, 2)  # (B, n_head, num_symbols, head_dim)
#         v_a = v.transpose(1, 2)  # (B, n_head, num_symbols, head_dim)

#         attn_out1 = F.scaled_dot_product_attention(
#             q_a, k_a, v_a,
#             dropout_p = self.dropout_p,
#             is_causal = self.is_causal
#         )  # → (B, n_head, T, head_dim)

#         # Back to (B, T, n_head, head_dim)
#         attn_out1 = attn_out1.permute(0, 2, 1, 3).contiguous()

#         # Residual #1 with uniform 1/K scaling
#         x_heads = x_heads + scale1 * attn_out1  # (B, T, n_head, head_dim)

#         # === 2) Second attention: “across heads” for each token ===
#         # Project x_heads → Q/K/V (shape (B, T, n_head, 3*head_dim))
#         qkv_h = self.qkv_proj_h(x_heads)
#         # Split into q_h, k_h, v_h each of shape (B, T, n_head, head_dim)
#         q_h, k_h, v_h = qkv_h.split(self.head_dim, dim=-1)

#         # Directly call attention treating T as num_heads and H as seq_len:
#         # q_h is (B, T, H, head_dim) → batch=B, num_heads=T, seq_len=H
#         attn_out2 = F.scaled_dot_product_attention(
#             q_h,        # (B, T, H, head_dim)
#             k_h,        # (B, T, H, head_dim)
#             v_h,        # (B, T, H, head_dim)
#             dropout_p = self.dropout_p,
#             is_causal = False
#         )  # → (B, T, H, head_dim)

#         # Residual #2 with uniform 1/K scaling
#         x_heads = x_heads + scale2 * attn_out2  # (B, T, H, head_dim)

#         # === 3) Optional Feed-Forward on each head-slice ===
#         if self.include_ffn:
#             ff_out  = self.ffn(self.ln2(x_heads))                 # (B, T, n_head, head_dim)
#             # Residual #3 with uniform 1/K scaling
#             x_heads = x_heads + scale3 * ff_out

#         # Restore shape to (B, T, d_model) and return
#         return x_heads.view(B, T, self.d_model)

@dataclass
class Config:
    # Perception
    n_layers: int = 6
    causal: bool = True

    # Computer
    n_symbols: int = 100
    include_ffn: bool = False
    shared_compute_block: bool = True
    causal_compute: bool = False  # Enable causal attention in compute blocks

    # Program

    # When n_prog_tokens is 0, the model is functionally equivalent to a pure decoder only Transformer,
    # Cross attention becomes a pass through identity.
    n_prog_tokens: int = 8 
    n_programs: int = 1

    # Common:
    d_model: int = 64
    n_heads: int = 4
    dropout: float = 0.0



class TransComputer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Handle n_prog_tokens=0 case
        if config.n_prog_tokens == 0:
            # Create embeddings for operation conditioning
            self.prog_embd = nn.Embedding(config.n_programs, config.d_model)  # Operation embeddings
            self.prog_pos_embd = nn.Embedding(1, config.d_model)  # Dummy
            self.register_buffer("pos_ids", torch.zeros(1, 1, dtype=torch.long), persistent=False)
        else:
            self.prog_embd = nn.Embedding(config.n_programs, config.d_model * config.n_prog_tokens)
            self.prog_pos_embd = nn.Embedding(config.n_prog_tokens, config.d_model)
            self.register_buffer("pos_ids", torch.arange(config.n_prog_tokens).unsqueeze(0), persistent=False)

        self.perception_blocks = nn.ModuleList([TransformerBlock(d_model=config.d_model, 
                                                      n_head=config.n_heads, 
                                                      dropout=config.dropout, 
                                                      is_causal=config.causal) for _ in range(config.n_layers)])
        
        # Only create compute and cross-attention blocks if we have program tokens
        if config.n_prog_tokens > 0:
            compute_blocks = [ComputeBlock(d_model=config.d_model, 
                                            n_head=config.n_heads, 
                                            dropout=config.dropout, 
                                            is_causal=config.causal_compute,
                                            num_symbols=config.n_symbols,
                                            include_ffn=config.include_ffn,
                                            causal_cross_head=config.causal_compute) for _ in range(config.n_layers)]
            
            perception_gxattns = [GatedCrossAttention(embed_dim=config.d_model, 
                                                    num_heads=config.n_heads, 
                                                    ff_hidden_dim=config.d_model * 4, 
                                                    dropout=config.dropout) for _ in range(config.n_layers)]
            
            compute_gxattns = [GatedCrossAttention(embed_dim=config.d_model, 
                                                    num_heads=config.n_heads, 
                                                    ff_hidden_dim=config.d_model * 4, 
                                                    dropout=config.dropout) for _ in range(config.n_layers)]
            
            self.perception_gated_xattn = nn.ModuleList([perception_gxattns[0]] * config.n_layers if config.shared_compute_block else perception_gxattns)
            self.compute_gated_xattn = nn.ModuleList([compute_gxattns[0]] * config.n_layers if config.shared_compute_block else compute_gxattns)
            self.compute_blocks = nn.ModuleList([compute_blocks[0]] * config.n_layers if config.shared_compute_block else compute_blocks)
        else:
            # Create empty ModuleLists for consistency
            self.perception_gated_xattn = nn.ModuleList()
            self.compute_gated_xattn = nn.ModuleList()
            self.compute_blocks = nn.ModuleList()
            
        self.ln_out = nn.LayerNorm(config.d_model)
        self.reset_parameters()

    def reset_parameters(self):
        self.prog_embd.weight.data.normal_(0, 0.02)
        self.prog_pos_embd.weight.data.normal_(0, 0.02)

    def get_key_padding_mask(self, input_lens, T: int):
        """
        input_lens: (B)
        T: sequence length of the input
        """

        # 1) create a position‐index vector of shape (PerL,)
        pos = torch.arange(T, device=input_lens.device).unsqueeze(0)   # shape (1, PerL)

        # 2) expand input_len to shape (B, 1) so it can broadcast against pos
        input_len_exp = input_lens.unsqueeze(1)  # shape (B, 1)

        # 3) compare: True wherever pos >= input_len_exp
        #    → key_padding_mask[b, t] = True  iff t >= input_len[b] 
        # This is because mask = True values are ignored by the cross attention.
        key_padding_mask = pos >= input_len_exp  # broadcasts to (B, PerL)
        return key_padding_mask

    def forward(self, x, prog_ids=None, input_lens=None, compute_steps: int = 1):
        """
        x: (B, T, D)
        prog_ids: (B)
        """
        if prog_ids is None:
            assert self.config.n_programs == 1, "n_programs must be 1 if prog_ids is None"
            prog_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if input_lens is None:
            assert self.config.n_programs == 1, "input_lens must be provided if n_programs > 1"
            input_lens = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Handle n_prog_tokens=0 case: pure transformer with operation conditioning
        if self.config.n_prog_tokens == 0:
            # Inject program embedding into all input tokens for operation conditioning
            prog = self.prog_embd(prog_ids)  # (B, d_model * n_prog_tokens)
            # Since n_prog_tokens=0, this gives us (B, d_model) - the operation embedding
            prog = prog.view(x.size(0), -1, self.config.d_model)  # (B, 1, d_model)
            
            # Add operation conditioning to all input tokens
            conditioned_x = x + prog  # (B, T, d_model) + (B, 1, d_model) -> (B, T, d_model)
            
            # Run perception blocks with conditioned input
            prev_h = conditioned_x
            for i in range(self.config.n_layers):
                prev_h = self.perception_blocks[i](prev_h)
            return self.ln_out(prev_h)

        # Standard TransComputer with program tokens
        prog = self.prog_embd(prog_ids) # (B, d_model * n_prog_tokens)
        prog = prog.view(x.size(0), -1, self.config.d_model) # (B, n_prog_tokens, d_model)

        # Add positional embeddings to program
        prog_pos = self.prog_pos_embd(self.pos_ids) # (1, n_prog_tokens, d_model)
        prog = prog + prog_pos

        prev_h = x
        prev_p = prog

        key_padding_mask = self.get_key_padding_mask(input_lens, x.size(1))

        for i in range(self.config.n_layers):
            next_h_i = self.perception_blocks[i](prev_h)
            # next_p_i = self.compute_blocks[i](prev_p)

            # Recursive compute block
            p = prev_p  # (B, N_prog, D)
            for _ in range(compute_steps):
                p = self.compute_blocks[i](p, compute_steps)
            next_p_i = p

            next_h = self.perception_gated_xattn[i](next_h_i, next_p_i)
            next_p = self.compute_gated_xattn[i](next_p_i, next_h_i, key_padding_mask=key_padding_mask)
            prev_h = next_h
            prev_p = next_p

        return self.ln_out(next_h)


if __name__ == "__main__":
    config = Config()
    model = TransComputer(config)

    batch_size = 32
    seq_len = 10
    
    prog_ids = torch.randint(0, config.n_programs, (batch_size,))
    x = torch.randn(batch_size, seq_len, config.d_model)


    model.forward(x, prog_ids)
    # model.forward(torch.randn(1, 10, 64), torch.randint(0, 100, (1, 10)))
