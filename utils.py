import torch
import torch.nn as nn
import math
from pathlib import Path
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from datasets import load_dataset

# From config.py
def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 150,
        "d_model": 512,
        "lang_src": "fi",
        "lang_tgt": "en",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "datasource": "local",
        "positional_encoding": "rope", # or "sinusoidal", "relative"
        "decoding_strategy": "greedy", # or "beam", "top-k"
        "beam_size": 5,
        "top_k": 3,
        "gdrive_path": "/content/drive/MyDrive/translation_anlp/"
    }

def get_weights_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    
    base_path = Path('.')
    gdrive_path = config.get('gdrive_path')
    if gdrive_path:
        base_path = Path(gdrive_path)
        
    return str(base_path / model_folder / model_filename)

# From model.py (shared layers)
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x: torch.Tensor):
        # x: (batch, h, seq_len, d_k)
        seq_len = x.shape[2]
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]
        
        # Reshape x to apply rotation to pairs of features
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_rotated = torch.stack(
            [
                x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin,
                x_reshaped[..., 0] * sin + x_reshaped[..., 1] * cos,
            ],
            dim=-1,
        )
        return x_rotated.flatten(-2)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float, rope: RotaryPositionalEmbedding = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = rope

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        if self.rope:
            query = self.rope(query)
            key = self.rope(key)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)

# From dataset.py
class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_tgt_pair = self.dataset[idx]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]
        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        enc_pad_len = self.seq_len - len(src_tokens) - 2
        dec_pad_len = self.seq_len - len(tgt_tokens) - 1
        if enc_pad_len < 0 or dec_pad_len < 0:
            raise ValueError("Sentence is too long")
        encoder_input = torch.cat([self.sos_token, torch.tensor(src_tokens, dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token] * enc_pad_len, dtype=torch.int64)], dim=0)
        decoder_input = torch.cat([self.sos_token, torch.tensor(tgt_tokens, dtype=torch.int64), torch.tensor([self.pad_token] * dec_pad_len, dtype=torch.int64)], dim=0)
        label = torch.cat([torch.tensor(tgt_tokens, dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token] * dec_pad_len, dtype=torch.int64)], dim=0)
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        return {"encoder_input": encoder_input, "decoder_input": decoder_input, "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), "label": label, "src_text": src_text, "tgt_text": tgt_text}

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

# From train.py (helpers)
def get_all_sentences(config, language):
    data_path = Path('data')
    file_path = data_path / f"EUbookshop.{language}"
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

def get_or_build_tokenizer(config, dataset, language):
    base_path = Path('.')
    gdrive_path = config.get('gdrive_path')
    if gdrive_path:
        base_path = Path(gdrive_path)
    
    tokenizer_path = base_path / config['tokenizer_file'].format(language)
    
    if not tokenizer_path.exists():
        from tokenizers.models import WordLevel
        from tokenizers.trainers import WordLevelTrainer
        from tokenizers.pre_tokenizers import Whitespace
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        
        # Ensure the directory for the tokenizer exists
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        
        tokenizer.train_from_iterator(get_all_sentences(config, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        decoder_output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(decoder_output[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        if next_word.item() == eos_idx:
            break
    return decoder_input.squeeze(0)
