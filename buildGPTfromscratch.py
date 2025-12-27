import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import gc
from pathlib import Path

# hyperparameters - M2チップ環境に最適化
batch_size = 64      # メモリ使用量削減のため128から64に削減
block_size = 128     # what is the maximum context length for predictions?
max_iters = 5001
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # M2チップ用にMPSを優先
eval_iters = 100     # メモリ使用量削減のため200から100に削減
n_embd = 128
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# データファイルの存在確認
input_file = 'input.txt'
if not os.path.exists(input_file):
    print(f"エラー: ファイル {input_file} が見つかりません。")
    print("input.txtファイルが正しいパスに存在することを確認してください。")
    exit(1)

# Load dataset
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if len(text) < 1000:
        print("警告: テキストファイルが小さすぎます。最低1000文字以上推奨します。")
        
except Exception as e:
    print(f"ファイル読み込みエラー: {e}")
    exit(1)

# Character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"語彙サイズ: {vocab_size}")
print(f"テキスト長: {len(text)}")

# Train/validation split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Batch data sampling
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Loss estimation
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

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, 16) @ (B, 16 , T) ----> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    "multiple heads of self-attention in parallel"

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head:the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Language model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)            # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb                                # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)                                     # final layer norm
        logits = self.lm_head(x)                             # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        model.eval()  # 生成時は評価モード
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]  # crop context if needed
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]        # (B,C)
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
                idx = torch.cat((idx, idx_next), dim=1)             # (B,T+1)
        model.train()  # 生成後は訓練モードに戻す
        return idx

# チェックポイント保存ディレクトリの作成
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Initialize model
print(f"デバイス: {device}")
model = BigramLanguageModel().to(device)

# モデルパラメータ数の表示
total_params = sum(p.numel() for p in model.parameters())
print(f"モデルパラメータ数: {total_params:,}")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

# Training loop
print("訓練開始...")
try:
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Save model checkpoint
            if iter > 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'iter': iter,
                    'losses': losses,
                    'vocab_size': vocab_size,
                    'n_embd': n_embd,
                    'n_head': n_head,
                    'n_layer': n_layer,
                    'block_size': block_size,
                    'stoi': stoi,
                    'itos': itos
                }
                checkpoint_path = checkpoint_dir / f'gpt_checkpoint_iter_{iter}.pt'
                torch.save(checkpoint, checkpoint_path)
                print(f"チェックポイント保存: {checkpoint_path}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # メモリクリア（M2チップ環境での安定性向上）
        if iter % 100 == 0:
            gc.collect()
            if device == 'mps':
                torch.mps.empty_cache()

except KeyboardInterrupt:
    print("\n訓練が中断されました。")
except Exception as e:
    print(f"訓練中にエラーが発生しました: {e}")
    raise

# Save final model
print("最終モデルを保存中...")
final_checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'iter': max_iters,
    'vocab_size': vocab_size,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'block_size': block_size,
    'stoi': stoi,
    'itos': itos
}
final_model_path = checkpoint_dir / 'gpt_final_model.pt'
torch.save(final_checkpoint, final_model_path)
print(f"最終モデル保存完了: {final_model_path}")

# Generate text with different temperatures
print("\n=== テキスト生成開始 ===")

# 安全な開始文字の選択
if len(text) > 0:
    start_char = text[0]
else:
    start_char = list(stoi.keys())[0]  # 語彙の最初の文字を使用

print(f"開始文字: '{start_char}'")

try:
    print("=== Generated Text (Temperature=1.0) ===")
    context = torch.tensor([[stoi[start_char]]], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500, temperature=1.0)
    print(decode(generated[0].tolist()))

    print("\n=== Generated Text (Temperature=0.8) ===")
    generated = model.generate(context, max_new_tokens=500, temperature=0.8)
    print(decode(generated[0].tolist()))

    print("\n=== Generated Text (Temperature=1.2) ===")
    generated = model.generate(context, max_new_tokens=500, temperature=1.2)
    print(decode(generated[0].tolist()))

except Exception as e:
    print(f"テキスト生成中にエラーが発生しました: {e}")

print("\n=== 処理完了 ===")
