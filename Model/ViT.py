from Encoder.ViT_Encoder import ViT
from Decoder.Transformer_Decoder import TransformerDecoder

import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, max_seq_len, vocab_size, patch_size = 10, num_patches = 100, enc_embedding_dim = 128, enc_num_layers = 12,
                 dec_embedding_dim = 128, dec_num_layers = 12, dropout=0.1):
        super().__init__()

        # ViT Encoder
        self.encoder = ViT(
            patch_size = patch_size,
            num_patches = num_patches,
            embedding_dim = enc_embedding_dim,
            num_layers = enc_num_layers
        )

        # Transformer Decoder
        self.decoder = TransformerDecoder(
            max_seq_len = max_seq_len,
            vocab_size = vocab_size,
            embed_dim = dec_embedding_dim,
            num_layers = dec_num_layers,
            dropout = dropout
          )

        # final output projection
        self.output_projection = nn.Linear(dec_embedding_dim, vocab_size)

    def forward(self, src, tgt):
        """
        src (source): (batch_size, channel, height, width)
        tgt (target): (batch_size, max_seq_len)
        """

        enc_out = self.encoder(src)
        dec_out, causal_attn, cross_attn = self.decoder(tgt, enc_out)
        logits = self.output_projection(dec_out)  # Shape: (batch_size, max_seq_len - 1, vocab_size)

        return logits
        #return logits, causal_attn, cross_attn
