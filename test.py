import torch
import sys
from pathlib import Path
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import build_transformer
from utils import get_config, greedy_decode, beam_search_decode, top_k_sampling_decode, BilingualDataset, load_dataset_by_split

def get_test_dataset(config):
    """Load and create the test dataset for evaluation using pre-defined splits"""
    # Load the pre-split test data
    test_data = load_dataset_by_split(config, 'test')

    # Load tokenizers
    tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt']))
    
    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))

    # Create test dataset and dataloader
    test_dataset = BilingualDataset(test_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Test dataset loaded: {len(test_dataset)} examples")

    return test_dataloader, tokenizer_src, tokenizer_tgt

def evaluate_model(epoch_number: str):
    """Evaluate the model on the test set and calculate BLEU score"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config = get_config()
    
    # Load tokenizers
    tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt']))
    
    if not tokenizer_src_path.exists() or not tokenizer_tgt_path.exists():
        print("Tokenizer files not found in current directory.")
        print(f"Expected: {tokenizer_src_path} and {tokenizer_tgt_path}")
        sys.exit(1)
        
    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
    
    # Build model
    model = build_transformer(
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size(), 
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model']
    ).to(device)

    # Load the pretrained weights from local weights folder
    model_filename = f"./weights/tmodel_{int(epoch_number):02d}.pt"
    print(f"Loading model weights from epoch {epoch_number}: {model_filename}")
    
    if not Path(model_filename).exists():
        print(f"Model checkpoint not found: {model_filename}")
        print("Available checkpoints in ./weights/:")
        weights_folder = Path("./weights")
        if weights_folder.exists():
            checkpoints = list(weights_folder.glob("tmodel_*.pt"))
            if checkpoints:
                for checkpoint in sorted(checkpoints):
                    print(f"  - {checkpoint.name}")
            else:
                print("  No checkpoints found.")
        else:
            print("  ./weights/ folder does not exist.")
        sys.exit(1)
    
    state = torch.load(model_filename, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])

    # Get test dataset
    test_dataloader, _, _ = get_test_dataset(config)
    
    print(f"Evaluating model on {len(test_dataloader)} test examples...")
    
    model.eval()
    predicted = []
    expected = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            # Use greedy decoding for evaluation
            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
            
            tgt_text = batch['tgt_text'][0]
            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            expected.append(tgt_text)
            predicted.append(model_output_text)

    # Calculate simple BLEU score manually (since torchmetrics might not be available)
    try:
        import torchmetrics
        bleu_metric = torchmetrics.text.BLEUScore()
        bleu = bleu_metric([p.lower() for p in predicted], [[e.lower()] for e in expected])
        
        print("=" * 50)
        print(f"EVALUATION RESULTS FOR EPOCH {epoch_number}")
        print("=" * 50)
        print(f"BLEU Score:  {bleu:.4f}")
        print("=" * 50)
        
    except ImportError:
        print("=" * 50)
        print(f"EVALUATION RESULTS FOR EPOCH {epoch_number}")
        print("=" * 50)
        print("torchmetrics not available, showing sample predictions:")
        print("=" * 50)
        
    # Show a few examples
    print("\nSample translations:")
    print("-" * 50)
    for i in range(min(5, len(predicted))):
        print(f"Expected:  {expected[i]}")
        print(f"Predicted: {predicted[i]}")
        print("-" * 50)

def translate(epoch_number: str, sentence: str):
    """
    Translates a sentence using greedy, beam search, and top-k sampling decoding strategies.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config = get_config()
    
    # Load tokenizers from local paths
    tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt']))
    
    if not tokenizer_src_path.exists() or not tokenizer_tgt_path.exists():
        print("Tokenizer files not found in current directory.")
        print(f"Expected: {tokenizer_src_path} and {tokenizer_tgt_path}")
        sys.exit(1)
        
    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
    
    # Build model
    model = build_transformer(
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size(), 
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model']
    ).to(device)

    # Load the pretrained weights from local weights folder
    model_filename = f"./weights/tmodel_{int(epoch_number):02d}.pt"
    print(f"Loading model weights from epoch {epoch_number}: {model_filename}")
    
    if not Path(model_filename).exists():
        print(f"Model checkpoint not found: {model_filename}")
        print("Available checkpoints in ./weights/:")
        weights_folder = Path("./weights")
        if weights_folder.exists():
            checkpoints = list(weights_folder.glob("tmodel_*.pt"))
            if checkpoints:
                for checkpoint in sorted(checkpoints):
                    print(f"  - {checkpoint.name}")
            else:
                print("  No checkpoints found.")
        else:
            print("  ./weights/ folder does not exist.")
        sys.exit(1)
    
    state = torch.load(model_filename, map_location=device, weights_only=False)
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
    print("=" * 50)
    print(f"TESTING EPOCH {epoch_number}")
    print("=" * 50)
    print(f"SOURCE:    {sentence}")
    print("-" * 50)
    print(f"Greedy:    {output_text_greedy}")
    print(f"Beam (k={config['beam_size']}): {output_text_beam}")
    print(f"Top-k (k={config['top_k']}):  {output_text_top_k}")
    print("=" * 50)

if __name__ == "__main__":
    # Check for evaluation mode
    if len(sys.argv) >= 3 and "--evaluate" in sys.argv:
        epoch_number = sys.argv[1]
        evaluate_model(epoch_number)
    elif len(sys.argv) >= 3:
        # Translation mode
        epoch_number = sys.argv[1]
        sentence_to_translate = " ".join(sys.argv[2:])
        translate(epoch_number, sentence_to_translate)
    else:
        print("Usage:")
        print("  Translation: python test.py <epoch_number> <sentence_to_translate>")
        print("  Evaluation:  python test.py <epoch_number> --evaluate")
        print("Examples:")
        print("  python test.py 1 'Tämä on testivirke.'")
        print("  python test.py 4 --evaluate")
