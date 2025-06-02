import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 40000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('data/toy_lm.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class EfficientMultiHeadAttention(nn.Module):
    """
    Computes multi‐head self‐attention more efficiently by projecting
    to Q/K/V for all heads in a single linear, then splitting and computing
    attention in parallel.
    """
    def __init__(self, d_model, n_head, block_size, dropout):
        """
        Args:
            n_embd    (int): total embedding dimension (must be divisible by n_head)
            n_head    (int): number of attention heads
            block_size(int): maximum sequence length (for building the causal mask)
            dropout  (float): dropout probability on attention weights & final proj
        """
        super().__init__()
        assert d_model % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_size = d_model // n_head
        self.total_head_size = n_head * self.head_size  # = n_embd

        # one linear to compute Q, K, V for *all* heads: output dim = 3 * n_embd
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # final projection back to n_embd
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # causal mask of shape (1, 1, block_size, block_size) so we can slice [ :T, :T ] later
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("causal_mask", mask == 0)  # True where j > i

    def forward(self, x):
        """
        x: (B, T, n_embd)
        returns: (B, T, n_embd)
        """
        B, T, C = x.size()  # C == n_embd
        # 1) project to QKV
        #    qkv: (B, T, 3 * n_embd)
        qkv = self.qkv_proj(x)

        # 2) reshape/split into (Q, K, V)
        #    first, reshape to (B, T, n_head, 3*head_size)
        qkv = qkv.view(B, T, self.n_head, 3 * self.head_size)
        #    then split last dim → q, k, v each (B, T, n_head, head_size)
        q, k, v = qkv.split(self.head_size, dim=-1)

        # 3) reorder to (B, n_head, T, head_size) for matmul
        #    i.e. .permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)  # (B, n_head, T, head_size)
        k = k.permute(0, 2, 1, 3)  # (B, n_head, T, head_size)
        v = v.permute(0, 2, 1, 3)  # (B, n_head, T, head_size)

        # 4) scaled dot‐product: (B, n_head, T, head_size) @ (B, n_head, head_size, T) → (B, n_head, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)

        # 5) apply causal mask: mask shape = (1, 1, block_size, block_size)
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T],
            float("-inf")
        )

        # 6) softmax → (B, n_head, T, T)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 7) weighted sum: (B, n_head, T, T) @ (B, n_head, T, head_size) → (B, n_head, T, head_size)
        attn_output = attn_probs @ v

        # 8) reorder back to (B, T, n_head, head_size) then merge heads → (B, T, n_embd)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(B, T, C)

        # 9) final linear + dropout
        out = self.out_proj(attn_output)
        out = self.dropout(out)
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, d_model, n_head, dropout: float = 0.0):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = d_model // n_head
        # self.sa = MultiHeadAttention(n_head, head_size)
        self.sa = EfficientMultiHeadAttention(d_model, n_head, block_size, dropout)
        self.ffwd = FeedFoward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(d_model=n_embd, n_head=n_head, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
# model = torch.compile(model, mode="reduce-overhead", fullgraph=True, backend="inductor")
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
