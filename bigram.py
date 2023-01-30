import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # How many independent sequences will we process in parallel?
block_size = 8  # What is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda'
eval_iters = 200
n_embed = 32  # Number of embeddings
# ------------
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

# Load dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create a mapping from characters to integers
strToNum = {ch: i for i, ch in enumerate(chars)}
numToStr = {i: ch for i, ch in enumerate(chars)}
encode = lambda string: [strToNum[char] for char in string]
decode = lambda num_list: ''.join([numToStr[i] for i in num_list])

# Separate data into train/test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # First 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    batch_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(batch_data) - block_size, (batch_size,))
    new_x = torch.stack([batch_data[i: i + block_size] for i in ix])
    new_y = torch.stack([batch_data[i + 1: i + block_size + 1] for i in ix])
    new_x, new_y = new_x.to(device), new_y.to(device)
    return new_x, new_y


@torch.no_grad()
def estimate_loss():
    out = {}
    bigram_model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            new_logits, new_loss = bigram_model(X, Y)
            losses[k] = new_loss.item()
        out[split] = losses.mean()
    bigram_model.train()
    return out


class BigramLanguageModel(nn.Module):
    token_embedding_table: nn.Embedding

    def __init__(self, ):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            nn.LayerNorm(n_embed),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        curr_b, curr_t = idx.shape

        # Idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(curr_t, device=device))  # (T, C)
        combined_x = tok_emb + pos_emb  # (B, T, C)
        combined_x = self.blocks(combined_x)  # (B, T, C)
        new_logits = self.lm_head(combined_x)  # (B, T, vocab_size)

        if targets is None:
            new_loss = None
        else:
            b, t, c = new_logits.shape
            new_logits = new_logits.view(b * t, c)
            targets = targets.view(b * t)
            new_loss = F.cross_entropy(new_logits, targets)

        return new_logits, new_loss

    def generate(self, idx, max_new_tokens):
        # Idx is a (B, T) array of indicies in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # Get the predictions
            new_logits, new_loss = self(idx_cond)

            # Focus only on the last time step
            new_logits = new_logits[:, -1, :]  # Get last element of the time dimension which is (B, C)

            # Apply softmax to get the probabilities
            probs = F.softmax(new_logits, dim=1)  # Becomes (B, C)

            # Sample 1 sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # Becomes (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)  # Becomes (B, T+1)
        return idx


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class FeedForward(nn.Module):
    """ A simple linear layer followed by non-linearity """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class Head(nn.Module):
    """ One head of self attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # B, T, T
        wei = F.softmax(wei, dim=-1)  # B, T, T

        # Perform weighted aggregation of the values
        v = self.value(x)  # B, T, C
        out = wei @ v  # B, T, C
        return out


bigram_model = BigramLanguageModel()
bigram_model = bigram_model.to(device)

optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=1e-3)

for iteration in range(max_iters):
    # Every once in a while evaluate the loss on train and val sets
    if iteration % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = bigram_model(xb, yb)

    # Evaluate the loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

test_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(bigram_model.generate(test_idx, max_new_tokens=100)[0].tolist()))
