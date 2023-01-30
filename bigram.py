import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # How many independent sequences will we process in parallel?
block_size = 8  # What is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda'
eval_iters = 200
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

    def __init__(self, input_vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        print(input_vocab_size)
        self.token_embedding_table = nn.Embedding(input_vocab_size, input_vocab_size)

    def forward(self, idx, targets=None):
        # Idx and targets are both (B, T) tensors of integers
        new_logits = self.token_embedding_table(idx)  # (B, T, C)

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
            # Get the predictions
            new_logits, new_loss = self(idx)

            # Focus only on the last time step
            new_logits = new_logits[:, -1, :]  # Get last element of the time dimension which is (B, C)

            # Apply softmax to get the probabilities
            probs = F.softmax(new_logits, dim=1)  # Becomes (B, C)

            # Sample 1 sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # Becomes (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)  # Becomes (B, T+1)
        return idx


bigram_model = BigramLanguageModel(vocab_size)
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
