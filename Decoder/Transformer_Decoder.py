import torch
import torch.nn as nn
import torch.nn.functional as F

class Sequence_Embedding(nn.Module):
    def __init__(self, max_seq_length: int, vocab_size: int, embed_dim: int):
        '''
        Class tỏch.nn.Embedding:
        num_embeddings (int) – size of the dictionary of embeddings
        embedding_dim (int) – the size of each embedding vector
        '''
        super().__init__()
        self.pos_embedding = nn.Embedding(num_embeddings = max_seq_length,
                                        embedding_dim = embed_dim)
        self.token_embedding = nn.Embedding(num_embeddings = vocab_size,
                                            embedding_dim = embed_dim)
    def forward(self, sequence: torch.tensor):
        # Input sequence's shape: (batch_size, seq_len)
        batch_size, seq_len = sequence.shape

        # Token embedding
        token_embed = self.token_embedding(sequence)         # (batch_size, seq_len, embed_dim)

        # Position embedding
        position = torch.arange(seq_len, device = sequence.device)  # (seq_len,)
        position = position.unsqueeze(0)                     # (1, seq_len)
        pos_embed = self.pos_embedding(position)             # (1, seq_len, embed_dim)

        return token_embed + pos_embed

class Causal_Multihead_Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0 # Make sure that embedding dim is divisable to number of heads

        self.layer_norm = nn.LayerNorm(normalized_shape = embed_dim,)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim = embed_dim,
                                                        num_heads = num_heads,
                                                        dropout = dropout,
                                                        batch_first = True,)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x: torch.tensor, token_ids: torch.tensor, pad_id):
        x_norm = self.layer_norm(x)
        max_seq_len = x.shape[1]

        # Create mask
        key_padding_mask = (token_ids == pad_id)    # shape (batch_size, tokens), dtype bool
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, device=x.device), diagonal=1).bool()
        output, attn_weights = self.multi_head_attention(query = x_norm,
                                                        key = x_norm,
                                                        value = x_norm,
                                                        attn_mask = causal_mask,
                                                        key_padding_mask = key_padding_mask)
        output = self.dropout(output)
        return x + output, attn_weights
    
class Cross_Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0 # Make sure that embedding dim is divisable to number of heads

        self.layer_norm = nn.LayerNorm(normalized_shape = embed_dim)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim = embed_dim,
                                                        num_heads = num_heads,
                                                        dropout = dropout,
                                                        batch_first = True)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x: torch.tensor, context: torch.tensor):
        # context vector is from the ViT encoder
        x_norm = self.layer_norm(x)
        output, attn_weights = self.multi_head_attention(query = x_norm,
                                                        key = context,
                                                        value = context,
                                                        is_causal = False)
        output = self.dropout(output)
        return x + output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape = embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features = embed_dim,
                    out_features = hidden_dim),
            nn.GELU(),
            nn.Dropout(p = dropout),
            nn.Linear(in_features = hidden_dim,
                    out_features = embed_dim),
            nn.Dropout(p = dropout)
        )
    def forward(self, x: torch.tensor):
        x_norm = self.layer_norm(x)
        return x + self.feed_forward(x_norm)
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffw_dim: int, max_seq_len, vocab_size, dropout: float = 0.1):
        super().__init__()
        self.causal_attention = Causal_Multihead_Attention(embed_dim, num_heads, dropout)
        self.cross_attention = Cross_Attention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ffw_dim, dropout)
    def forward(self, x: torch.tensor, token_ids: torch.tensor, context: torch.tensor):
        x, causal_attn_weights = self.causal_attention(x, token_ids)
        x, cross_attn_weights = self.cross_attention(x, context)
        x = self.feed_forward(x)
        return x, causal_attn_weights, cross_attn_weights
    
class TransformerDecoder(nn.Module):
    def __init__(self, max_seq_len, vocab_size, embed_dim: int, num_layers: int, num_heads: int = 16, ffw_dim: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.seq_embedding = Sequence_Embedding(max_seq_len, vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ffw_dim, max_seq_len, vocab_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, token_ids: torch.tensor, context: torch.tensor):
        causal_attn_weights = None
        cross_attn_weights = None
        x = self.seq_embedding(token_ids)
        for layer in self.layers:
            x, causal_attn_weights, cross_attn_weights = layer(x, token_ids, context)
        return x, causal_attn_weights, cross_attn_weights