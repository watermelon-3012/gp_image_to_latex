import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import json
import os

# 1. CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_HEIGHT = 50
IMG_WIDTH = 200
PATCH_SIZE = 10

EMBED_DIM = 128
MAX_SEQ_LEN = 253  # Must match training data shape (was 152, checkpoint uses 253)

MAX_FILE_SIZE = 5 * 1024 * 1024

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 2. LOAD VOCAB
try:
    vocab_path = 'vocab.json'
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    itos = {v: k for k, v in vocab.items()}
    stoi = vocab
    PAD_IDX = stoi.get('<pad>', 1)
    SOS_IDX = stoi.get('<sos>', 2)
    EOS_IDX = stoi.get('<eos>', 3)
    print(f"[OK] Vocab loaded! Size: {len(vocab)}")

except Exception as e:
    print(f"[ERROR] Error loading vocab: {e}")
    stoi = {'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3}
    itos = {0: '<unk>', 1: '<pad>', 2: '<sos>', 3: '<eos>'}
    PAD_IDX, SOS_IDX, EOS_IDX = 1, 2, 3

max_seq_len = MAX_SEQ_LEN

# 3. MODEL 1: CNN-LSTM
class CnnEncoder(nn.Module):
    def __init__(self, input_shape, embedding_dim):
        input_channel = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            test = torch.randn(1, input_channel, height, width)
            output = self.features(test)
            flatten_size = output.view(output.shape[0], -1).shape[1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.features(x)
        out = self.fc(out)
        return out

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size=512, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM input: embedding_dim + encoder_output_dim (128)
        self.lstm = nn.LSTM(embedding_dim + 128, hidden_size, num_layers, batch_first=True)
        # IMPORTANT: Use fc_out to match the checkpoint layer name
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim

    def forward(self, encoder_output, decoder_input):
        embedded = self.embedding(decoder_input)  # (B, seq_len, embedding_dim)
        batch_size, seq_len, _ = embedded.shape
        # Repeat encoder output for each timestep
        encoder_repeated = encoder_output.unsqueeze(1).expand(-1, seq_len, -1)  # (B, seq_len, 128)
        lstm_input = torch.cat([embedded, encoder_repeated], dim=2)  # (B, seq_len, embedding_dim + 128)
        output, _ = self.lstm(lstm_input)
        output = self.fc_out(output)
        return output

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_shape, embedding_dim, vocab_size):
        super().__init__()
        self.encoder = CnnEncoder(input_shape, embedding_dim)
        self.decoder = LSTMDecoder(vocab_size, embedding_dim, hidden_size=512, num_layers=1)

    def forward(self, image_input, decoder_input_ids):
        encoder_output = self.encoder(image_input)
        decoder_output = self.decoder(encoder_output, decoder_input_ids)
        return decoder_output

    def generate(self, img, max_len=80):
        self.eval()
        with torch.no_grad():
            encoder_out = self.encoder(img)
            seq = [SOS_IDX]
            for _ in range(max_len):
                inp = torch.LongTensor([seq]).to(DEVICE)
                out = self.decoder(encoder_out, inp)
                next_token = out[0, -1, :].argmax().item()
                if next_token == EOS_IDX or next_token == PAD_IDX:
                    break
                seq.append(next_token)
            return seq[1:]

# 4. MODEL 2: ViT (Vision Transformer)
def extract_patches(image, patch_size):
    B, C, H, W = image.shape
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.view(B, -1, patch_size * patch_size)
    return patches

class Patch_embedding_ViT(nn.Module):
    def __init__(self, patch_size, num_patches, embedding_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.patch_embed = nn.Linear(patch_size**2, embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))

    def forward(self, x):
        B, N, P = x.shape
        patch_embedding = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, patch_embedding), dim=1)
        return tokens + self.pos_embed

class MultiheadAttention_ViT(nn.Module):
    def __init__(self, embedding_dim, num_head=16, dropout=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_head, dropout=dropout, batch_first=True)

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x_attention, _ = self.mha(query=x_norm, key=x_norm, value=x_norm, need_weights=False)
        return x + x_attention

class MLP_Block_ViT(nn.Module):
    def __init__(self, embedding_dim, mlp_size=4096, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return x + self.mlp(x_norm)

class TransformerEncoderLayer_ViT(nn.Module):
    def __init__(self, embedding_dim, num_head=16, attn_dropout=0, mlp_size=4096, mlp_dropout=0.1):
        super().__init__()
        self.attn = MultiheadAttention_ViT(embedding_dim, num_head, attn_dropout)
        self.mlp = MLP_Block_ViT(embedding_dim, mlp_size, mlp_dropout)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x

class TransformerEncoder_ViT(nn.Module):
    def __init__(self, embedding_dim, num_layers, num_head=16, attn_dropout=0, mlp_size=4096, mlp_dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer_ViT(embedding_dim, num_head, attn_dropout, mlp_size, mlp_dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ViT(nn.Module):
    def __init__(self, patch_size, num_patches, embedding_dim, num_layers):
        super().__init__()
        self.patch_size = patch_size
        self.embed = Patch_embedding_ViT(patch_size, num_patches, embedding_dim)
        self.transformer = TransformerEncoder_ViT(embedding_dim, num_layers, num_head=16, mlp_size=4096)

    def forward(self, x):
        out1 = extract_patches(x, self.patch_size)
        out2 = self.embed(out1)
        out3 = self.transformer(out2)
        return out3

# 5. MODEL 3: CvT (Convolutional Vision Transformer)
class Cnn_layer(nn.Module):
    def __init__(self, input_channel, height, width, embedding_dim, max_len=5000):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(input_channel, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout2d(0.2)
        )
        self.projection = nn.Linear(128, embedding_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embedding_dim))

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        return x

class MultiheadAttention_CvT(nn.Module):
    def __init__(self, embedding_dim, num_head=4, dropout=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_head, dropout=dropout, batch_first=True)

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x_attention, _ = self.mha(query=x_norm, key=x_norm, value=x_norm, need_weights=False)
        return x + x_attention

class MLP_Block_CvT(nn.Module):
    def __init__(self, embedding_dim, mlp_size=1024, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return x + self.mlp(x_norm)

class TransformerEncoderLayer_CvT(nn.Module):
    def __init__(self, embedding_dim, num_head=4, attn_dropout=0, mlp_size=1024, mlp_dropout=0.1):
        super().__init__()
        self.attn = MultiheadAttention_CvT(embedding_dim, num_head, attn_dropout)
        self.mlp = MLP_Block_CvT(embedding_dim, mlp_size, mlp_dropout)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x

class TransformerEncoder_CvT(nn.Module):
    def __init__(self, embedding_dim, num_layers, num_head=4, attn_dropout=0, mlp_size=1024, mlp_dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer_CvT(embedding_dim, num_head, attn_dropout, mlp_size, mlp_dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class CvT(nn.Module):
    def __init__(self, input_channel, height, width, embedding_dim, num_layer):
        super().__init__()
        self.cnn = Cnn_layer(input_channel, height, width, embedding_dim)
        self.transformer = TransformerEncoder_CvT(embedding_dim, num_layer, num_head=4, mlp_size=1024)

    def forward(self, x):
        out1 = self.cnn(x)
        out2 = self.transformer(out1)
        return out2

# 6. SHARED DECODER FOR ViT & CvT (Updated to match notebook)
class Sequence_Embedding(nn.Module):
    def __init__(self, max_seq_length: int, vocab_size: int, embed_dim: int):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_embeddings=max_seq_length, embedding_dim=embed_dim)
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

    def forward(self, sequence):
        batch_size, seq_len = sequence.shape
        token_embed = self.token_embedding(sequence)
        position = torch.arange(seq_len, device=sequence.device).unsqueeze(0)
        pos_embed = self.pos_embedding(position)
        return token_embed + pos_embed

class Causal_Multihead_Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, token_ids, pad_id=None):
        if pad_id is None:
            pad_id = PAD_IDX
        x_norm = self.layer_norm(x)
        max_seq_len = x.shape[1]
        key_padding_mask = (token_ids == pad_id)
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, device=x.device), diagonal=1).bool()
        output, attn_weights = self.multi_head_attention(
            query=x_norm, key=x_norm, value=x_norm,
            attn_mask=causal_mask, key_padding_mask=key_padding_mask
        )
        output = self.dropout(output)
        return x + output, attn_weights

class Cross_Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, context):
        x_norm = self.layer_norm(x)
        output, attn_weights = self.multi_head_attention(
            query=x_norm, key=context, value=context, is_causal=False
        )
        output = self.dropout(output)
        return x + output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=embed_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return x + self.feed_forward(x_norm)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffw_dim: int, max_seq_len, vocab_size, dropout: float = 0.1):
        super().__init__()
        self.causal_attention = Causal_Multihead_Attention(embed_dim, num_heads, dropout)
        self.cross_attention = Cross_Attention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ffw_dim, dropout)

    def forward(self, x, token_ids, context):
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

    def forward(self, token_ids, context):
        causal_attn_weights = None
        cross_attn_weights = None
        x = self.seq_embedding(token_ids)
        for layer in self.layers:
            x, causal_attn_weights, cross_attn_weights = layer(x, token_ids, context)
        return x, causal_attn_weights, cross_attn_weights

# 7. FULL MODELS: ViT_Model & CvT_Model (Updated to match notebook - Transformer class)
class ViT_Model(nn.Module):
    def __init__(self, patch_size=10, num_patches=100, enc_embedding_dim=128, enc_num_layers=12,
                 dec_embedding_dim=128, dec_num_layers=12, dropout=0.1):
        super().__init__()
        # ViT Encoder
        self.encoder = ViT(
            patch_size=patch_size,
            num_patches=num_patches,
            embedding_dim=enc_embedding_dim,
            num_layers=enc_num_layers
        )
        # Transformer Decoder (new signature: max_seq_len, vocab_size, embed_dim, ...)
        self.decoder = TransformerDecoder(
            max_seq_len=MAX_SEQ_LEN,
            vocab_size=len(vocab),
            embed_dim=dec_embedding_dim,
            num_layers=dec_num_layers,
            dropout=dropout
        )
        self.output_projection = nn.Linear(dec_embedding_dim, len(vocab))

    def forward(self, src, tgt):
        enc_out = self.encoder(src)
        dec_out, causal_attn, cross_attn = self.decoder(tgt, enc_out)
        logits = self.output_projection(dec_out)
        return logits

    def generate(self, img, max_len=80, temperature=0.7):
        self.eval()
        with torch.no_grad():
            mem = self.encoder(img)
            seq = torch.LongTensor([[SOS_IDX]]).to(DEVICE)
            generated_tokens = []
            for _ in range(max_len):
                dec_out, _, _ = self.decoder(seq, mem)
                logits = self.output_projection(dec_out)[:, -1, :] / temperature
                token = logits.argmax(dim=-1).item()
                if token in [EOS_IDX, PAD_IDX]:
                    break
                generated_tokens.append(token)
                seq = torch.cat([seq, torch.LongTensor([[token]]).to(DEVICE)], dim=1)
            return generated_tokens

class CvT_Model(nn.Module):
    def __init__(self, input_channel=1, height=IMG_HEIGHT, width=IMG_WIDTH,
                 enc_embedding_dim=128, enc_num_layer=12,
                 dec_embedding_dim=128, dec_num_layers=12, dropout=0.1):
        super().__init__()
        self.encoder = CvT(input_channel, height, width, enc_embedding_dim, enc_num_layer)
        # Transformer Decoder (new signature: max_seq_len, vocab_size, embed_dim, ...)
        self.decoder = TransformerDecoder(
            max_seq_len=MAX_SEQ_LEN,
            vocab_size=len(vocab),
            embed_dim=dec_embedding_dim,
            num_layers=dec_num_layers,
            dropout=dropout
        )
        self.output_projection = nn.Linear(dec_embedding_dim, len(vocab))

    def forward(self, src, tgt):
        enc_out = self.encoder(src)
        dec_out, causal_attn, cross_attn = self.decoder(tgt, enc_out)
        logits = self.output_projection(dec_out)
        return logits

    def generate(self, img, max_len=80, temperature=0.7):
        self.eval()
        with torch.no_grad():
            mem = self.encoder(img)
            seq = torch.LongTensor([[SOS_IDX]]).to(DEVICE)
            generated_tokens = []
            for _ in range(max_len):
                dec_out, _, _ = self.decoder(seq, mem)
                logits = self.output_projection(dec_out)[:, -1, :] / temperature
                token = logits.argmax(dim=-1).item()
                if token in [EOS_IDX, PAD_IDX]:
                    break
                generated_tokens.append(token)
                seq = torch.cat([seq, torch.LongTensor([[token]]).to(DEVICE)], dim=1)
            return generated_tokens


# 8. LOAD ALL MODELS
models = {}

# Load CNN-LSTM
try:
    model_cnn_lstm = CNN_LSTM_Model(input_shape=[1, IMG_HEIGHT, IMG_WIDTH], embedding_dim=EMBED_DIM, vocab_size=len(vocab)).to(DEVICE)
    state_dict = torch.load('model_CNNLSTM.pth', map_location=DEVICE)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_cnn_lstm.load_state_dict(state_dict, strict=True)
    model_cnn_lstm.eval()
    models['cnn_lstm'] = model_cnn_lstm
    print("[OK] CNN-LSTM model loaded!")
except Exception as e:
    print(f"[ERROR] CNN-LSTM: {e}")
    models['cnn_lstm'] = None

# Load ViT
try:
    model_vit = ViT_Model().to(DEVICE)
    state_dict = torch.load('model_ViT_FORNEXTTRAINING.pth', map_location=DEVICE)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_vit.load_state_dict(state_dict, strict=True)
    model_vit.eval()
    models['vit'] = model_vit
    print("[OK] ViT model loaded!")
except Exception as e:
    print(f"[ERROR] ViT: {e}")
    models['vit'] = None

# Load CvT
try:
    model_cvt = CvT_Model().to(DEVICE)
    state_dict = torch.load('model_CvT_FORNEXTTRAINING.pth', map_location=DEVICE)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_cvt.load_state_dict(state_dict, strict=True)
    model_cvt.eval()
    models['cvt'] = model_cvt
    print("[OK] CvT model loaded!")
except Exception as e:
    print(f"[ERROR] CvT: {e}")
    models['cvt'] = None

# 9. HELPER FUNCTIONS
# Preprocessing pipeline - EXACT SAME as training
preprocess_pipeline = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((50, 200), antialias=True),  # (height, width)
    transforms.ToTensor(),  # Normalize to [0, 1]
])

def preprocess_image(image_bytes):
    """
    Chuyển bất kỳ ảnh nào thành tensor chuẩn như training:
    - Grayscale (1 channel)
    - Resize về (50, 200) - height x width
    - Normalize về [0, 1] via ToTensor()
    Output shape: (1, 1, 50, 200) - (batch, channel, height, width)
    """
    try:
        # Mở ảnh từ bytes
        img = Image.open(io.BytesIO(image_bytes))

        # Xử lý ảnh có alpha channel (RGBA, LA, PA) -> RGB với background trắng
        if img.mode in ('RGBA', 'LA', 'PA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])
            else:
                background.paste(img, mask=img.split()[1])
            img = background

        # Áp dụng pipeline preprocessing giống hệt training
        tensor = preprocess_pipeline(img)  # Shape: (1, 50, 200)

        # Debug info
        print(f"[DEBUG] Input image mode: {img.mode if hasattr(img, 'mode') else 'tensor'}")
        print(f"[DEBUG] Tensor shape: {tensor.shape}, min: {tensor.min():.4f}, max: {tensor.max():.4f}")

        return tensor.unsqueeze(0).to(DEVICE)  # Shape: (1, 1, 50, 200)
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def is_valid_image(file):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    if not file:
        return False
    filename = file.filename.lower()
    return '.' in filename and filename.rsplit('.', 1)[1] in allowed_extensions

def tokens_to_latex(tokens):
    return "".join([
        itos.get(i, '')
        for i in tokens
        if itos.get(i, '') not in ['<sos>', '<eos>', '<pad>', '<unk>']
    ])

# BEAM SEARCH FUNCTIONS
import torch.nn.functional as F

def beam_search_cnn_lstm(model, image, max_len=80, beam_size=5):
    """Beam search for CNN-LSTM model."""
    model.eval()
    with torch.no_grad():
        encoder_out = model.encoder(image)

        # Initialize beams: (score, sequence)
        beams = [(0.0, [SOS_IDX])]

        for _ in range(max_len):
            candidates = []

            for score, seq in beams:
                if seq[-1] == EOS_IDX or seq[-1] == PAD_IDX:
                    candidates.append((score, seq))
                    continue

                inp = torch.LongTensor([seq]).to(DEVICE)
                out = model.decoder(encoder_out, inp)
                logits = out[0, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)

                top_probs, top_ids = torch.topk(log_probs, beam_size)

                for k in range(beam_size):
                    new_score = score + top_probs[k].item()
                    new_seq = seq + [top_ids[k].item()]
                    # Length normalization
                    norm_score = new_score / len(new_seq)
                    candidates.append((norm_score, new_seq))

            # Keep top beam_size candidates
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

            # Early stop if all beams ended
            if all(seq[-1] in [EOS_IDX, PAD_IDX] for _, seq in beams):
                break

        # Return best sequence (remove SOS)
        best_seq = beams[0][1]
        return [t for t in best_seq if t not in [SOS_IDX, EOS_IDX, PAD_IDX]]

def beam_search_transformer(model, image, max_len=80, beam_size=5):
    """Beam search for ViT and CvT models."""
    model.eval()
    with torch.no_grad():
        mem = model.encoder(image)

        # Initialize beams: (score, sequence)
        beams = [(0.0, [SOS_IDX])]

        for _ in range(max_len):
            candidates = []

            for score, seq in beams:
                if seq[-1] == EOS_IDX or seq[-1] == PAD_IDX:
                    candidates.append((score, seq))
                    continue

                inp = torch.LongTensor([seq]).to(DEVICE)
                dec_out, _, _ = model.decoder(inp, mem)
                logits = model.output_projection(dec_out)[:, -1, :]
                log_probs = F.log_softmax(logits[0], dim=-1)

                top_probs, top_ids = torch.topk(log_probs, beam_size)

                for k in range(beam_size):
                    new_score = score + top_probs[k].item()
                    new_seq = seq + [top_ids[k].item()]
                    # Length normalization
                    norm_score = new_score / len(new_seq)
                    candidates.append((norm_score, new_seq))

            # Keep top beam_size candidates
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

            # Early stop if all beams ended
            if all(seq[-1] in [EOS_IDX, PAD_IDX] for _, seq in beams):
                break

        # Return best sequence (remove SOS)
        best_seq = beams[0][1]
        return [t for t in best_seq if t not in [SOS_IDX, EOS_IDX, PAD_IDX]]

# 10. FLASK ROUTES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'vocab_size': len(vocab),
        'models': {
            'cnn_lstm': models['cnn_lstm'] is not None,
            'vit': models['vit'] is not None,
            'cvt': models['cvt'] is not None
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file found'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not is_valid_image(file):
            return jsonify({'error': 'Invalid image format'}), 400

        image_bytes = file.read()
        if len(image_bytes) > MAX_FILE_SIZE:
            return jsonify({'error': 'File too large (max 5MB)'}), 400

        tensor = preprocess_image(image_bytes)

        results = {}

        # CNN-LSTM prediction (using greedy decoding)
        if models['cnn_lstm']:
            try:
                pred = models['cnn_lstm'].generate(tensor, max_len=80)
                results['cnn_lstm'] = {
                    'name': 'CNN-LSTM',
                    'latex': tokens_to_latex(pred),
                    'status': 'success'
                }
            except Exception as e:
                results['cnn_lstm'] = {'name': 'CNN-LSTM', 'latex': '', 'status': 'error', 'error': str(e)}
        else:
            results['cnn_lstm'] = {'name': 'CNN-LSTM', 'latex': '', 'status': 'not_loaded'}

        # ViT prediction (using greedy decoding)
        if models['vit']:
            try:
                pred = models['vit'].generate(tensor, max_len=80)
                results['vit'] = {
                    'name': 'Vision Transformer (ViT)',
                    'latex': tokens_to_latex(pred),
                    'status': 'success'
                }
            except Exception as e:
                results['vit'] = {'name': 'Vision Transformer (ViT)', 'latex': '', 'status': 'error', 'error': str(e)}
        else:
            results['vit'] = {'name': 'Vision Transformer (ViT)', 'latex': '', 'status': 'not_loaded'}

        # CvT prediction (using greedy decoding)
        if models['cvt']:
            try:
                pred = models['cvt'].generate(tensor, max_len=80)
                results['cvt'] = {
                    'name': 'Convolutional Vision Transformer (CvT)',
                    'latex': tokens_to_latex(pred),
                    'status': 'success'
                }
            except Exception as e:
                results['cvt'] = {'name': 'Convolutional Vision Transformer (CvT)', 'latex': '', 'status': 'error', 'error': str(e)}
        else:
            results['cvt'] = {'name': 'Convolutional Vision Transformer (CvT)', 'latex': '', 'status': 'not_loaded'}

        return jsonify({'success': True, 'results': results})

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large (max 5MB)'}), 413

# 11. RUN SERVER
if __name__ == '__main__':
    print("\n" + "="*60)
    print("MATH FORMULA OCR - MULTI MODEL COMPARISON")
    print("="*60)
    print(f"URL: http://localhost:5000")
    print(f"Device: {DEVICE}")
    print(f"Vocab size: {len(vocab)}")
    print("-"*60)
    print("Models loaded:")
    print(f"  - CNN-LSTM: {'OK' if models['cnn_lstm'] else 'FAILED'}")
    print(f"  - ViT:      {'OK' if models['vit'] else 'FAILED'}")
    print(f"  - CvT:      {'OK' if models['cvt'] else 'FAILED'}")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
