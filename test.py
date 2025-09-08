import torch
import sys
from pathlib import Path
from tokenizers import Tokenizer
from model import build_transformer
from utils import get_config, get_weights_path, greedy_decode

def translate(sentence: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config = get_config()
    
    # Load tokenizers
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    
    # Build model
    model = build_transformer(
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size(), 
        config["seq_len"], 
        config['seq_len'], 
        d_model=config['d_model']
    ).to(device)

    # Load the pretrained weights
    # Update the epoch number to the best model
    model_filename = get_weights_path(config, "19") 
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    seq_len = config['seq_len']

    model.eval()
    with torch.no_grad():
        # Encode the source sentence
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source = source.unsqueeze(0)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(1).int().to(device)

        # Decode the sentence
        model_out = greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, seq_len, device)
        output_text = tokenizer_tgt.decode(model_out.tolist())

    print(f"Source: {sentence}")
    print(f"Predicted: {output_text}")

if __name__ == "__main__":
    # Read sentence from command line argument
    if len(sys.argv) > 1:
        sentence_to_translate = sys.argv[1]
        translate(sentence_to_translate)
    else:
        print("Please provide a sentence to translate, for example: python test.py 'Hello world'")
