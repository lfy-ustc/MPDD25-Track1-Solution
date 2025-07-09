from models.networks.multihead_attention import CrossAttention
import torch.nn as nn

class TransformerCrossAttentionBlock(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerCrossAttentionBlock, self).__init__()
        self.cross_attn = CrossAttention(in_dim1, in_dim2, k_dim, v_dim, num_heads)
        self.norm1 = nn.LayerNorm(in_dim1)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim1, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, in_dim1)
        )
        self.norm2 = nn.LayerNorm(in_dim1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, mask=None):
        attn_output = self.cross_attn(x1, x2, mask=mask)
        x1 = self.norm1(x1 + self.dropout(attn_output))
        ffn_output = self.ffn(x1)
        x1 = self.norm2(x1 + self.dropout(ffn_output))
        return x1
        
class TransformerWithCrossAttention(nn.Module):
    def __init__(self, dim1, dim2, k_dim, v_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TransformerWithCrossAttention, self).__init__()
        self.layers = nn.ModuleList([
            TransformerCrossAttentionBlock(dim1, dim2, k_dim, v_dim, num_heads, ff_dim,dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim1)
    
    def forward(self, x1, x2, mask=None):
        for layer in self.layers:
            x1 = layer(x1, x2, mask)
        return self.norm(x1)
