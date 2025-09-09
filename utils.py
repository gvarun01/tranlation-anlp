import torch
import torch.nn as nn
import math
import json
import random
from pathlib import Path
from torch.utils.data import Dataset
from tokenizers import Tokenizer

def get_model_module(model):
    """Helper function to get the model module, handling DataParallel wrapper"""
    return model.module if hasattr(model, 'module') else model

# From config.py
def get_config():
    return {
        "batch_size": 64,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 150,
        "d_model": 512,
        "N": 6,  # Number of transformer layers
        "h": 8,  # Number of attention heads
        "dropout": 0.1,  # Dropout rate
        "d_ff": 2048,  # Feed-forward dimension
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
        "warmup_steps": 2000,
        "early_stopping_patience": 5,  # Stop if no improvement for N epochs
        "gradient_accumulation_steps": 1,  # Accumulate gradients over N steps
        "use_amp": True,  # Use automatic mixed precision
        "compile_model": True,  # Use torch.compile for speed
        "val_eval_frequency": 1,  # Evaluate validation every N epochs
        "save_frequency": 1  # Save checkpoint every N epochs
    }

def get_weights_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    
    base_path = Path('.')
    return str(base_path / model_folder / model_filename)

def find_latest_checkpoint(config):
    """
    Find the latest checkpoint in the working directory or Kaggle input.
    Returns (checkpoint_path, epoch_number) or (None, None) if no checkpoint found.
    """
    model_basename = config['model_basename']
    
    # Check local working directory first
    local_weights_dir = Path('.') / config['model_folder']
    if local_weights_dir.exists():
        local_checkpoints = list(local_weights_dir.glob(f"{model_basename}*.pt"))
        if local_checkpoints:
            # Find the latest checkpoint by epoch number
            latest_local = max(local_checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            epoch_num = int(latest_local.stem.split('_')[-1])
            print(f"Found latest local checkpoint: {latest_local}")
            return str(latest_local), epoch_num
    
    # Check Kaggle input directory
    kaggle_input_dir = Path('/kaggle/input')
    if kaggle_input_dir.exists():
        # Look for any translation-related input directories
        for input_dir in kaggle_input_dir.iterdir():
            if input_dir.is_dir() and 'translation' in input_dir.name.lower():
                kaggle_checkpoints = list(input_dir.glob(f"{model_basename}*.pt"))
                if kaggle_checkpoints:
                    latest_kaggle = max(kaggle_checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
                    epoch_num = int(latest_kaggle.stem.split('_')[-1])
                    print(f"Found latest Kaggle checkpoint: {latest_kaggle}")
                    return str(latest_kaggle), epoch_num
    
    print("No existing checkpoints found. Starting training from scratch.")
    return None, None

def get_checkpoint_path_for_epoch(config, epoch_number):
    """
    Get checkpoint path for a specific epoch. Searches in working directory first, then Kaggle input.
    Special case: 'best' loads tmodel_best.pt
    """
    model_basename = config['model_basename']
    
    # Special case for 'best' model
    if epoch_number == 'best':
        checkpoint_filename = f"{model_basename}best.pt"
    else:
        try:
            epoch_str = f"{int(epoch_number):02d}"
        except ValueError:
            print(f"Invalid epoch number: {epoch_number}. Expected an integer or 'best'.")
            return None
        checkpoint_filename = f"{model_basename}{epoch_str}.pt"
    
    # Check local working directory first
    local_path = Path('.') / config['model_folder'] / checkpoint_filename
    if local_path.exists():
        return str(local_path)
    
    # Check Kaggle input directory
    kaggle_input_dir = Path('/kaggle/input')
    if kaggle_input_dir.exists():
        for input_dir in kaggle_input_dir.iterdir():
            if input_dir.is_dir() and 'translation' in input_dir.name.lower():
                kaggle_path = input_dir / checkpoint_filename
                if kaggle_path.exists():
                    return str(kaggle_path)
    
    # Return None if not found
    return None

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
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (max_seq_len, dim/2)
        cos = freqs.cos()
        sin = freqs.sin()
        # Interleave to match even/odd dims
        cos = torch.stack([cos, cos], dim=-1).reshape(max_seq_len, dim)
        sin = torch.stack([sin, sin], dim=-1).reshape(max_seq_len, dim)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1)
        return x_rot.flatten(-2)

    def forward(self, x: torch.Tensor):
        # x: (batch, h, seq_len, d_k)
        seq_len = x.shape[-2]
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,dim)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        return (x * cos) + (self.rotate_half(x) * sin)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float, rope: RotaryPositionalEmbedding = None, attn_bias_module: nn.Module = None) -> None:
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
        self.attn_bias_module = attn_bias_module

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout, bias: torch.Tensor = None):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Use dtype-aware minimum to avoid fp16 overflow on masked_fill
            fill_value = torch.finfo(attention_scores.dtype).min
            attention_scores.masked_fill_(mask == 0, fill_value)
        if bias is not None:
            # Ensure bias matches device and dtype
            attention_scores = attention_scores + bias.to(device=attention_scores.device, dtype=attention_scores.dtype)
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

        # Compute relative position bias if configured
        bias = None
        if self.attn_bias_module is not None:
            # mask shape can be (batch, 1, Lq, Lk) or (batch, Lq) & causal elsewhere; we only need lengths
            Lq = query.shape[-2]
            Lk = key.shape[-2]
            # Disable AMP for relative position bias computation to avoid indexing issues
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.attn_bias_module(Lq, Lk)  # (1, h, Lq, Lk)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout, bias)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)

class RelativePositionBias(nn.Module):
    """
    T5/Shaw-style relative position bias producing an additive attention bias per head.
    """
    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        # T5-style signed buckets
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        n = -relative_position
        sign = (n > 0).to(torch.long)
        half = num_buckets // 2
        n = n.abs()
        # exact for small positions
        is_small = n < half
        val_if_small = n
        # logarithmic buckets for larger positions
        # avoid log(0)
        n_clamped = torch.clamp(n, min=1)
        log_ratio = torch.log(n_clamped.float() / half) / math.log(max_distance / half)
        val_if_large = (log_ratio * (num_buckets - half)).floor().to(torch.long) + half
        val_if_large = torch.clamp(val_if_large, max=num_buckets - 1)
        val = torch.where(is_small, val_if_small, val_if_large)
        # fold sign: first half for negative, second half for positive
        result = val + sign * half
        # Ensure all values are within bounds
        result = torch.clamp(result, 0, num_buckets - 1)
        return result

    def forward(self, q_len: int, k_len: int) -> torch.Tensor:
        device = self.relative_attention_bias.weight.device
        context_position = torch.arange(q_len, device=device)[:, None]
        memory_position = torch.arange(k_len, device=device)[None, :]
        relative_position = memory_position - context_position  # (q_len, k_len)
        rp_bucket = self._relative_position_bucket(relative_position)
        
        # Debug: Check for out-of-bounds indices
        if torch.any(rp_bucket < 0) or torch.any(rp_bucket >= self.num_buckets):
            print(f"Warning: Out-of-bounds indices in relative position bias!")
            print(f"Min: {rp_bucket.min()}, Max: {rp_bucket.max()}, Expected range: [0, {self.num_buckets-1}]")
            rp_bucket = torch.clamp(rp_bucket, 0, self.num_buckets - 1)
        
        # (q_len, k_len, num_heads)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, q_len, k_len)
        return values

# From dataset.py
def causal_mask(size, device=None):
    mask = torch.triu(torch.ones((1, size, size), device=device), diagonal=1).type(torch.int)
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
    model_module = get_model_module(model)
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    encoder_output = model_module.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1), device=device).type_as(source_mask).to(device)
        decoder_output = model_module.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model_module.project(decoder_output[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        if next_word.item() == eos_idx:
            break
    return decoder_input.squeeze(0)

def beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, beam_size):
    model_module = get_model_module(model)
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model_module.encode(source, source_mask)

    # Start with a single beam containing only the SOS token
    beams = [(torch.tensor([sos_idx], dtype=torch.long, device=device), 0.0)]

    for _ in range(max_len):
        new_beams = []
        all_ended = True
        for seq, score in beams:
            if seq[-1].item() == eos_idx:
                new_beams.append((seq, score))
                continue
            
            all_ended = False

            decoder_input = seq.unsqueeze(0)
            decoder_mask = causal_mask(decoder_input.size(1), device=device).type_as(source_mask).to(device)
            
            decoder_output = model_module.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            
            prob = model_module.project(decoder_output[:, -1])
            log_prob = torch.log_softmax(prob, dim=-1)
            
            top_k_log_probs, top_k_indices = torch.topk(log_prob, beam_size, dim=-1)

            for i in range(beam_size):
                new_seq = torch.cat([seq, top_k_indices[0, i].unsqueeze(0)], dim=0)
                new_score = score + top_k_log_probs[0, i].item()
                new_beams.append((new_seq, new_score))

        # Sort all new beams by score and keep the top `beam_size`
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        # Check if all beams have ended with EOS
        if all_ended:
            break

    # Return the sequence from the beam with the highest score
    best_seq, _ = beams[0]
    return best_seq

def top_k_sampling_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, top_k):
    model_module = get_model_module(model)
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model_module.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1), device=device).type_as(source_mask).to(device)
        decoder_output = model_module.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        prob = model_module.project(decoder_output[:, -1])
        
        # Apply top-k sampling
        top_k_probs, top_k_indices = torch.topk(prob, top_k, dim=-1)
        
        # Create a new distribution from the top-k probabilities
        dist = torch.distributions.categorical.Categorical(logits=top_k_probs)
        next_token_idx_in_top_k = dist.sample()
        next_word = top_k_indices.gather(-1, next_token_idx_in_top_k.unsqueeze(-1)).squeeze(-1)

        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)

        if next_word.item() == eos_idx:
            break
            
    return decoder_input.squeeze(0)

def create_dataset_splits(config, force_recreate=False):
    """
    Create and save dataset splits (train/val/test) to ensure consistency across runs.
    
    Args:
        config: Configuration dictionary
        force_recreate: If True, recreate splits even if they exist
    
    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    splits_file = Path('./dataset_splits.json')
    
    # Check if splits already exist and we don't want to recreate
    if splits_file.exists() and not force_recreate:
        print(f"Loading existing dataset splits from {splits_file}")
        with open(splits_file, 'r') as f:
            splits_data = json.load(f)
        return splits_data['train_indices'], splits_data['val_indices'], splits_data['test_indices']
    
    print("Creating new dataset splits...")
    
    # Load and filter the dataset (same logic as before)
    src_lang = config['lang_src']
    tgt_lang = config['lang_tgt']
    data_path = Path('data')
    src_path = data_path / f"EUbookshop.{src_lang}"
    tgt_path = data_path / f"EUbookshop.{tgt_lang}"

    with open(src_path, 'r', encoding='utf-8') as f:
        src_lines = f.read().splitlines()
    with open(tgt_path, 'r', encoding='utf-8') as f:
        tgt_lines = f.read().splitlines()

    # Create dataset in the expected format
    dataset_raw = []
    for src, tgt in zip(src_lines, tgt_lines):
        dataset_raw.append({
            'translation': {
                src_lang: src,
                tgt_lang: tgt
            }
        })

    # Get tokenizers to filter by length
    tokenizer_src = get_or_build_tokenizer(config, None, src_lang)
    tokenizer_tgt = get_or_build_tokenizer(config, None, tgt_lang)

    # Filter dataset by sequence length
    filtered_indices = []
    for i, item in enumerate(dataset_raw):
        src_len = len(tokenizer_src.encode(item['translation'][config['lang_src']]).ids)
        tgt_len = len(tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids)
        
        if src_len <= config['seq_len'] - 2 and tgt_len <= config['seq_len'] - 1:
            filtered_indices.append(i)
    
    # Set random seed for reproducible splits
    random.seed(42)
    random.shuffle(filtered_indices)
    
    # Calculate split sizes
    total_size = len(filtered_indices)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Create splits
    train_indices = filtered_indices[:train_size]
    val_indices = filtered_indices[train_size:train_size + val_size]
    test_indices = filtered_indices[train_size + val_size:]
    
    # Save splits to file
    splits_data = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'total_examples': total_size,
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices),
        'config_used': {
            'seq_len': config['seq_len'],
            'lang_src': config['lang_src'],
            'lang_tgt': config['lang_tgt']
        }
    }
    
    with open(splits_file, 'w') as f:
        json.dump(splits_data, f, indent=2)
    
    print(f"Dataset splits created and saved to {splits_file}")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return train_indices, val_indices, test_indices

def load_dataset_by_split(config, split_type='train'):
    """
    Load a specific split of the dataset.
    
    Args:
        config: Configuration dictionary
        split_type: 'train', 'val', or 'test'
    
    Returns:
        list: Dataset items for the specified split
    """
    # Get the splits
    train_indices, val_indices, test_indices = create_dataset_splits(config)
    
    # Choose the right indices based on split_type
    if split_type == 'train':
        indices = train_indices
    elif split_type == 'val':
        indices = val_indices
    elif split_type == 'test':
        indices = test_indices
    else:
        raise ValueError(f"Invalid split_type: {split_type}. Must be 'train', 'val', or 'test'")
    
    # Load the full dataset
    src_lang = config['lang_src']
    tgt_lang = config['lang_tgt']
    data_path = Path('data')
    src_path = data_path / f"EUbookshop.{src_lang}"
    tgt_path = data_path / f"EUbookshop.{tgt_lang}"

    with open(src_path, 'r', encoding='utf-8') as f:
        src_lines = f.read().splitlines()
    with open(tgt_path, 'r', encoding='utf-8') as f:
        tgt_lines = f.read().splitlines()

    # Create the full dataset
    dataset_raw = []
    for src, tgt in zip(src_lines, tgt_lines):
        dataset_raw.append({
            'translation': {
                src_lang: src,
                tgt_lang: tgt
            }
        })
    
    # Return only the items for this split
    split_data = [dataset_raw[i] for i in indices]
    return split_data

def causal_mask(size, device=None):
    mask = torch.triu(torch.ones((1, size, size), device=device), diagonal=1).type(torch.int)
    return mask == 0

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len, device=None):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.device = device

        # Source language specials (for encoder side)
        self.sos_src = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64, device=device)
        self.eos_src = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64, device=device)
        self.pad_src = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64, device=device)

        # Target language specials (for decoder side and labels)
        self.sos_tgt = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64, device=device)
        self.eos_tgt = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64, device=device)
        self.pad_tgt = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64, device=device)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_src,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_src,
                torch.tensor([self.pad_src] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_tgt,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_tgt] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_tgt,
                torch.tensor([self.pad_tgt] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_src).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_tgt).unsqueeze(0).int() & causal_mask(decoder_input.size(0), device=None), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
