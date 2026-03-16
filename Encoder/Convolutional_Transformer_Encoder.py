import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder.ViT_Encoder import TransformerEncoder

class Cnn_layer(nn.Module):
    def __init__(self, input_channel, height, width, embedding_dim ,max_len = 5000 ):
        super().__init__()
        self.feature = nn.Sequential(
            #Block 1
            nn.Conv2d(input_channel, 8, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            #Block 2
            nn.Conv2d(8, 16, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            #Block 3
            nn.Conv2d(16, 32, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            #Block 4
            nn.Conv2d(32, 64, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Dropout2d(0.2),
            #Block 5
            nn.Conv2d(64, 128, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Dropout2d(0.2)
        )
        self.projection = nn.Linear(128, embedding_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embedding_dim))
    def forward(self,x):
    # B, C, H, W
        x = self.feature(x)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(0,2,1)
        x = self.projection(x)
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]

        return x

class CvT(nn.Module):
    def __init__(self,  input_channel, height, width, embedding_dim, num_layer):
        super().__init__()
        self.cnn = Cnn_layer(input_channel, height, width, embedding_dim)

        self.transformer = TransformerEncoder(embedding_dim ,num_layer, num_head=4, attn_dropout=0, mlp_size=1024, mlp_dropout=0.1)
    def forward(self,x):
        out1 = self.cnn(x)
        out2 = self.transformer(out1)
        return out2