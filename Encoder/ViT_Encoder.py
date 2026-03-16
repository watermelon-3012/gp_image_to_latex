import torch
import torch.nn as nn
import torch.nn.functional as F

#patch_size = 10
def extract_patches(image, patch_size):
    B, C, H, W = image.shape # Batch, Channel, Height, Width
    assert H % patch_size == 0
    assert W % patch_size == 0
    patchs = image.unfold (2,patch_size,patch_size).unfold(3,patch_size,patch_size)
    patchs = patchs.contiguous().view(B, C, -1, patch_size, patch_size)
    B_new, C_new, N, first_patch_size, second_patch_size = patchs.shape
    patchs = patchs.view(B_new ,N , -1)
    return patchs

class Patch_embedding(nn.Module):
    def  __init__(self,patch_size, num_patchs, embedding_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patchs
        self.embedding_dim = embedding_dim
        self.patch_embed = nn.Linear(self.patch_size**2,self.embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embedding_dim))
    def forward(self,x):
        B, N , P = x.shape
        patch_embedding = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, patch_embedding), dim=1)
        return tokens + self.pos_embed
    
class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, num_head:int=16 , dropout: float = 0) -> None:
        super().__init__()
        self.layer_norm=nn.LayerNorm(embedding_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim,
                                        num_heads=num_head,
                                        dropout=dropout,
                                        batch_first=True)
    def forward(self,x):
        x_norm = self.layer_norm(x)
        x_attention,_ = self.mha(query = x_norm,
                            key = x_norm,
                            value = x_norm,
                            need_weights = False)
        return  x + x_attention

class MLP_Block(nn.Module):
    def __init__(self, embedding_dim,mlp_size: int = 4096 ,dropout: float = 0.1) :
        super().__init__()
        self.layer_norm=nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(nn.Linear(embedding_dim,mlp_size),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(mlp_size,embedding_dim),
                                nn.Dropout(dropout))
    def forward(self,x):
        x_norm = self.layer_norm(x)
        x_mlp = self.mlp(x_norm)
        return x + x_mlp

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_head:int=16 , attn_dropout: float = 0 , mlp_size: int =4096 , mlp_dropout: float = 0.1 ):
        super().__init__()
        self.attn = MultiheadAttention(embedding_dim,num_head,attn_dropout)
        self.mlp = MLP_Block(embedding_dim,mlp_size,mlp_dropout)
    def forward(self,x):
        x_attn = self.attn(x)
        x_mlp = self.mlp(x_attn)
        return x_mlp
    
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim,num_layers, num_head:int=16 , attn_dropout: float = 0 , mlp_size: int =4096 , mlp_dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
                TransformerEncoderLayer(
                    embedding_dim=embedding_dim,
                    num_head=num_head,
                    mlp_size=mlp_size,
                    attn_dropout=attn_dropout,
                    mlp_dropout=mlp_dropout
                )
                for _ in range(num_layers)
            ])
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ViT(nn.Module):
    def __init__(self, patch_size, num_patches, embedding_dim, num_layers):
        super().__init__()
        self.patch_size = patch_size
        self.embed = Patch_embedding(patch_size,num_patches,embedding_dim)
        self.transformer = TransformerEncoder(embedding_dim,num_layers)
    def forward(self,x):
        out1 = extract_patches(x,self.patch_size)
        out2 = self.embed(out1)
        out3 = self.transformer(out2)
        return out3