import torch
import sys
from pathlib import Path
from tokenizers import Tokenizer
from model import build_transformer
from utils import get_config, get_weights_path, greedy_decode, beam_search_decode, top_k_sampling_decode

def translate(sentence: str):
    """
    Translates a sentence using greedy, beam search, and top-k sampling decoding strategies.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config = get_config()
    
    # Load tokenizers from the path specified in the config
    base_path = Path('.')
    gdrive_path = config.get('gdrive_path')
    if gdrive_path and Path(gdrive_path).exists():
        base_path = Path(gdrive_path)
    
    tokenizer_src_path = base_path / config['tokenizer_file'].format(config['lang_src'])
    tokenizer_tgt_path = base_path / config['tokenizer_file'].format(config['lang_tgt'])
    
    if not tokenizer_src_path.exists() or not tokenizer_tgt_path.exists():
        print("Tokenizer files not found. Please run preprocess.py first.")
        sys.exit(1)
        
    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
    
    # Build model
    model = build_transformer(
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size(), 
        config["seq_len"], 
        config['d_model']
    ).to(device)

    # Load the pretrained weights
    # Update the epoch number to the best model if needed
    model_filename = get_weights_path(config, "19") 
    print(f"Loading model weights from: {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    seq_len = config['seq_len']

    model.eval()
    with torch.no_grad():
        # Preprocess the source sentence
        source_tokens = tokenizer_src.encode(sentence).ids
        pad_len = seq_len - len(source_tokens) - 2
        if pad_len < 0:
            print(f"Sentence is too long. Max length is {seq_len - 2}.")
            sys.exit(1)

        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(source_tokens, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * pad_len, dtype=torch.int64)
        ], dim=0).to(device)
        
        source = source.unsqueeze(0)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(2).int().to(device)

        # --- DECODING ---
        # Greedy Decode
        model_out_greedy = greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, seq_len, device)
        output_text_greedy = tokenizer_tgt.decode(model_out_greedy.tolist())

        # Beam Search Decode
        model_out_beam = beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, seq_len, device, config['beam_size'])
        output_text_beam = tokenizer_tgt.decode(model_out_beam.tolist())

        # Top-k Sampling
        model_out_top_k = top_k_sampling_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, seq_len, device, config['top_k'])
        output_text_top_k = tokenizer_tgt.decode(model_out_top_k.tolist())

    # --- PRINT RESULTS ---
    print("-" * 30)
    print(f"SOURCE:    {sentence}")
    print("-" * 30)
    print(f"Greedy:    {output_text_greedy}")
    print(f"Beam (k={config['beam_size']}): {output_text_beam}")
    print(f"Top-k (k={config['top_k']}):  {output_text_top_k}")
    print("-" * 30)

if __name__ == "__main__":
    # Read sentence from command line argument
    if len(sys.argv) > 1:
        sentence_to_translate = " ".join(sys.argv[1:])
        translate(sentence_to_translate)
    else:
        print("Please provide a sentence to translate.")
        print("Example: python test.py 'This is a test sentence.'")
