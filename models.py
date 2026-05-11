import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        # x 的输入形状是 Q: (batch_size, num_heads, seq_len, head_dim)
        seq_len = x.shape[2] 
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq) # (seq_len, d_model/2)
        emb = torch.cat((freqs, freqs), dim=-1) # (seq_len, d_model)
        
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)
def apply_rotary_pos_emb(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    # cos, sin: (1, 1, seq_len, head_dim)
    d = x.shape[-1]
    
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    cos_half = cos[..., :d//2]
    sin_half = sin[..., :d//2] 
    
    # 应用旋转公式
    return torch.cat([x1 * cos_half - x2 * sin_half, 
                      x1 * sin_half + x2 * cos_half], dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return self.pe[:, :x.size(1), :]

class AttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, max_seq_len=200, use_pe=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.use_pe = use_pe
        if use_pe:
            self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1) # 二分类，输出1个值配合BCEWithLogits

    def forward(self, x):
        x = self.embed(x)
        if self.use_pe:
            x = x + self.pos_encoder(x)
        
        # MHA with residual connection
        attn_out, _ = self.attn(x, x, x)
        x = self.layer_norm(x + attn_out)
        
        # 按照 README 要求，取最后一个 token 的表征
        x = x[:, -1, :] 
        return self.classifier(x).squeeze(-1)

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embed(x)
        output, _ = self.rnn(x)
        # 取最后一个 token
        out = output[:, -1, :]
        return self.classifier(out).squeeze(-1)

# --- 附加实验2：手动实现 MHA ---
class ManualMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads,use_rope=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        if use_rope:
            self.rope_embedding = RotaryPositionalEmbedding(self.head_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rope_embedding(Q) # 计算旋转角度
            Q = apply_rotary_pos_emb(Q, cos, sin) # 对 Q 进行旋转
            K = apply_rotary_pos_emb(K, cos, sin) # 对 K 进行旋转
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_out)

class ManualAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, max_seq_len=200,pe_type='absolute'):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attn = ManualMultiheadAttention(embed_dim, num_heads,use_rope=pe_type)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)
        self.pe_type = pe_type
        if pe_type == 'absolute':
            self.pos_encoder = PositionalEncoding(embed_dim) # 绝对位置编码
        else:
            self.pos_encoder = nn.Identity() # RoPE 模式下不加绝对位置编码

    def forward(self, x):
        x = self.embed(x)
        if self.pe_type == 'absolute':
            x = self.pos_encoder(x)+x
        attn_out = self.attn(x)
        x = self.layer_norm(x + attn_out)
        x = x[:, -1, :] 
        return self.classifier(x).squeeze(-1)