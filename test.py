import torch
import sys
from pathlib import Path
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import build_transformer
from utils import get_config, greedy_decode, beam_search_decode, top_k_sampling_decode, BilingualDataset, load_dataset_by_split, find_latest_checkpoint, get_checkpoint_path_for_epoch
try:
    import torchmetrics
    from bert_score import BERTScorer
    # BLEU score
    bleu_metric = torchmetrics.text.BLEUScore()
    
    # BERTScore
    # Use a pre-trained model for BERTScore, like 'roberta-large'
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

except ImportError:
    torchmetrics = None
    bert_scorer = None
    bleu_metric = None
    print("torchmetrics or bert_score not available, showing sample predictions only.")
    print("Please install with: pip install torchmetrics bert-score")

def get_test_dataset(config):
    """Load and create the test dataset for evaluation using pre-defined splits"""
    # Load the pre-split test data
    test_data = load_dataset_by_split(config, 'test')

    # Use tiny subset for debugging if specified
    if config.get('tiny_subset', False):
        print("Using tiny subset for testing...")
        test_data = test_data[:50]  # Use only 50 examples for testing
        print(f"Tiny subset - Test: {len(test_data)} examples")

    # Load tokenizers
    tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt']))
    
    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))

    # Create test dataset and dataloader
    test_dataset = BilingualDataset(test_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"Test dataset loaded: {len(test_dataset)} examples")

    return test_dataloader, tokenizer_src, tokenizer_tgt

def evaluate_model(epoch_number: str, config):
    """Evaluate the model on the test set and calculate BLEU score"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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

    # Load the pretrained weights using auto-detection
    model_filename = get_checkpoint_path_for_epoch(config, epoch_number)
    if model_filename:
        print(f"Loading model weights from epoch {epoch_number}: {model_filename}")
        try:
            state = torch.load(model_filename, map_location=device, weights_only=False)
            model.load_state_dict(state['model_state_dict'])
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
    else:
        print(f"Model checkpoint for epoch {epoch_number} not found!")
        print("Available checkpoints:")
        
        # Show available checkpoints
        weights_folder = Path("./weights")
        if weights_folder.exists():
            local_checkpoints = list(weights_folder.glob("tmodel_*.pt"))
            if local_checkpoints:
                print("  Local directory:")
                for checkpoint in sorted(local_checkpoints):
                    epoch_num = checkpoint.stem.split('_')[-1]
                    print(f"    - Epoch {epoch_num}: {checkpoint.name}")
        
        kaggle_input_dir = Path('/kaggle/input')
        if kaggle_input_dir.exists():
            for input_dir in kaggle_input_dir.iterdir():
                if input_dir.is_dir() and 'translation' in input_dir.name.lower():
                    kaggle_checkpoints = list(input_dir.glob("tmodel_*.pt"))
                    if kaggle_checkpoints:
                        print(f"  Kaggle input ({input_dir.name}):")
                        for checkpoint in sorted(kaggle_checkpoints):
                            epoch_num = checkpoint.stem.split('_')[-1]
                            print(f"    - Epoch {epoch_num}: {checkpoint.name}")
        
        sys.exit(1)

    # Get test dataset
    test_dataloader, _, _ = get_test_dataset(config)
    
    print(f"Evaluating model on {len(test_dataloader)} test examples...")
    
    model.eval()
    predicted_greedy = []
    predicted_beam = []
    predicted_top_k = []
    expected = []

    # Setup metrics
    try:
        import torchmetrics
        from bert_score import BERTScorer
        bleu_metric = torchmetrics.text.BLEUScore()
        bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device=device)
        metrics_available = True
    except ImportError:
        metrics_available = False
        print("\nWarning: torchmetrics or bert-score not installed. Skipping BLEU and BERTScore calculation.")
        print("Install them with: pip install torchmetrics bert-score")

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            encoder_input = batch['encoder_input'].to(device)  # Shape: (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)    # Shape: (batch_size, 1, seq_len, seq_len)
            
            batch_size = encoder_input.size(0)
            batch_greedy = []
            batch_beam = []
            batch_top_k = []
            
            # Process each example in the batch individually
            for i in range(batch_size):
                single_encoder_input = encoder_input[i:i+1]  # Shape: (1, seq_len)
                single_encoder_mask = encoder_mask[i:i+1]    # Shape: (1, 1, seq_len, seq_len)
                
                # Greedy Decode
                output_greedy = greedy_decode(model, single_encoder_input, single_encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
                batch_greedy.append(output_greedy)
                
                # Beam Search Decode
                output_beam = beam_search_decode(model, single_encoder_input, single_encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device, config['beam_size'])
                batch_beam.append(output_beam)
                
                # Top-k Sampling Decode
                output_top_k = top_k_sampling_decode(model, single_encoder_input, single_encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device, config['top_k'])
                batch_top_k.append(output_top_k)
            
            tgt_texts = batch['tgt_text']
            expected.extend(tgt_texts)
            
            # Decode all predictions
            model_output_texts_greedy = [tokenizer_tgt.decode(output.detach().cpu().numpy()) for output in batch_greedy]
            model_output_texts_beam = [tokenizer_tgt.decode(output.detach().cpu().numpy()) for output in batch_beam]
            model_output_texts_top_k = [tokenizer_tgt.decode(output.detach().cpu().numpy()) for output in batch_top_k]
            
            predicted_greedy.extend(model_output_texts_greedy)
            predicted_beam.extend(model_output_texts_beam)
            predicted_top_k.extend(model_output_texts_top_k)

    # Save predictions and expected translations to files
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_filepath_greedy = output_dir / f"predictions_greedy_epoch_{epoch_number}.txt"
    pred_filepath_beam = output_dir / f"predictions_beam_epoch_{epoch_number}.txt"
    pred_filepath_top_k = output_dir / f"predictions_top_k_epoch_{epoch_number}.txt"
    exp_filepath = output_dir / f"expected_epoch_{epoch_number}.txt"

    with open(pred_filepath_greedy, "w", encoding="utf-8") as f:
        f.write("\n".join(predicted_greedy))
    with open(pred_filepath_beam, "w", encoding="utf-8") as f:
        f.write("\n".join(predicted_beam))
    with open(pred_filepath_top_k, "w", encoding="utf-8") as f:
        f.write("\n".join(predicted_top_k))
    
    with open(exp_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(expected))
        
    print(f"Predictions saved to:")
    print(f"  Greedy: {pred_filepath_greedy}")
    print(f"  Beam:   {pred_filepath_beam}")
    print(f"  Top-k:  {pred_filepath_top_k}")
    print(f"Expected: {exp_filepath}")

    print("=" * 50)
    print(f"EVALUATION RESULTS FOR EPOCH {epoch_number}")
    print("=" * 50)

    if metrics_available:
        # Calculate BLEU and BERTScore for all decoding methods
        print("\n--- GREEDY DECODING ---")
        bleu_greedy = bleu_metric([p.lower() for p in predicted_greedy], [[e.lower()] for e in expected])
        print(f"BLEU Score:      {bleu_greedy:.4f}")
        P, R, F1 = bert_scorer.score(predicted_greedy, expected)
        bert_f1_greedy = F1.mean()
        print(f"BERTScore (F1):  {bert_f1_greedy:.4f}")

        print("\n--- BEAM SEARCH DECODING ---")
        bleu_beam = bleu_metric([p.lower() for p in predicted_beam], [[e.lower()] for e in expected])
        print(f"BLEU Score:      {bleu_beam:.4f}")
        P, R, F1 = bert_scorer.score(predicted_beam, expected)
        bert_f1_beam = F1.mean()
        print(f"BERTScore (F1):  {bert_f1_beam:.4f}")

        print("\n--- TOP-K SAMPLING DECODING ---")
        bleu_top_k = bleu_metric([p.lower() for p in predicted_top_k], [[e.lower()] for e in expected])
        print(f"BLEU Score:      {bleu_top_k:.4f}")
        P, R, F1 = bert_scorer.score(predicted_top_k, expected)
        bert_f1_top_k = F1.mean()
        print(f"BERTScore (F1):  {bert_f1_top_k:.4f}")
        
        print("\n" + "=" * 50)
    else:
        print("Metrics not available, showing sample predictions only.")
        print("=" * 50)
        
    # Show a few examples from greedy decoding
    print("\nSample translations (Greedy decoding):")
    print("-" * 50)
    for i in range(min(5, len(predicted_greedy))):
        print(f"Expected:  {expected[i]}")
        print(f"Predicted: {predicted_greedy[i]}")
        print("-" * 50)

def translate(epoch_number: str, sentence: str, config):
    """
    Translates a sentence using greedy, beam search, and top-k sampling decoding strategies.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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

    # Load the pretrained weights using auto-detection
    model_filename = get_checkpoint_path_for_epoch(config, epoch_number)
    if model_filename:
        print(f"Loading model weights from epoch {epoch_number}: {model_filename}")
        try:
            state = torch.load(model_filename, map_location=device, weights_only=False)
            model.load_state_dict(state['model_state_dict'])
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
    else:
        print(f"Model checkpoint for epoch {epoch_number} not found!")
        print("Available checkpoints:")
        
        # Show available checkpoints
        weights_folder = Path("./weights")
        if weights_folder.exists():
            local_checkpoints = list(weights_folder.glob("tmodel_*.pt"))
            if local_checkpoints:
                print("  Local directory:")
                for checkpoint in sorted(local_checkpoints):
                    epoch_num = checkpoint.stem.split('_')[-1]
                    print(f"    - Epoch {epoch_num}: {checkpoint.name}")
        
        kaggle_input_dir = Path('/kaggle/input')
        if kaggle_input_dir.exists():
            for input_dir in kaggle_input_dir.iterdir():
                if input_dir.is_dir() and 'translation' in input_dir.name.lower():
                    kaggle_checkpoints = list(input_dir.glob("tmodel_*.pt"))
                    if kaggle_checkpoints:
                        print(f"  Kaggle input ({input_dir.name}):")
                        for checkpoint in sorted(kaggle_checkpoints):
                            epoch_num = checkpoint.stem.split('_')[-1]
                            print(f"    - Epoch {epoch_num}: {checkpoint.name}")
        
        sys.exit(1)

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
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the transformer model")
    parser.add_argument("epoch_number", help="Epoch number to load")
    parser.add_argument("sentence", nargs='?', help="Sentence to translate (if not evaluating)")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation instead of translation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--tiny_subset", action="store_true", help="Use tiny subset for testing (same as training)")
    
    args = parser.parse_args()
    
    config = get_config()
    config['batch_size'] = args.batch_size
    config['tiny_subset'] = args.tiny_subset
    
    if args.evaluate:
        evaluate_model(args.epoch_number, config)
    elif args.sentence:
        translate(args.epoch_number, args.sentence, config)
    else:
        print("Usage:")
        print("  Translation: python test.py <epoch_number> <sentence_to_translate> [--batch_size BATCH_SIZE] [--tiny_subset]")
        print("  Evaluation:  python test.py <epoch_number> --evaluate [--batch_size BATCH_SIZE] [--tiny_subset]")
        print("Examples:")
        print("  python test.py 1 'Tämä on testivirke.' --batch_size 16")
        print("  python test.py 4 --evaluate --batch_size 32 --tiny_subset")
