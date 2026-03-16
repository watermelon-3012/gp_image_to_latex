import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder.Convolutional_Transformer_Encoder import CvT
from Decoder.Transformer_Decoder import TransformerDecoder

class Convolutional_Transformer(nn.Module):
  def __init__(self, input_channel, height, width, max_seq_len, vocab_size, enc_embedding_dim = 128, enc_num_layer = 12,
               dec_embedding_dim = 128, dec_num_layers = 12, dropout=0.1):
    super().__init__()
    self.encoder = CvT(input_channel, height, width, enc_embedding_dim, enc_num_layer)
    self.decoder = TransformerDecoder(
            max_seq_len = max_seq_len,
            vocab_size = vocab_size,
            embed_dim = dec_embedding_dim,
            num_layers = dec_num_layers,
            dropout = dropout
          )

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