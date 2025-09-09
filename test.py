import torch
import sys
from pathlib import Path
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import build_transformer
from utils import get_config, get_model_module, greedy_decode, beam_search_decode, top_k_sampling_decode, greedy_decode_optimized, beam_search_decode_optimized, top_k_sampling_decode_optimized, batch_greedy_decode, BilingualDataset, load_dataset_by_split, find_latest_checkpoint, get_checkpoint_path_for_epoch
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

    # Load tokenizers
    tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt']))
    
    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))

    # Create test dataset and dataloader
    test_dataset = BilingualDataset(test_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'], device=None)  # CPU for data loading
    # Use optimized settings for multi-GPU evaluation
    num_workers = min(4, config.get('num_workers', 4))  # Limit workers for Kaggle
    pin_memory = torch.cuda.is_available()  # Pin memory for faster GPU transfer
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"Test dataset loaded: {len(test_dataset)} examples")

    return test_dataloader, tokenizer_src, tokenizer_tgt, len(test_dataset)

def evaluate_model(epoch_number: str, config):
    """Evaluate the model on the test set and calculate BLEU score"""
    # Check for GPU availability and count
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        requested_gpus = config.get('num_gpus', available_gpus)
        if requested_gpus is None:
            requested_gpus = available_gpus
        num_gpus = min(requested_gpus, available_gpus)
        device = torch.device('cuda')
        print(f"Found {available_gpus} GPU(s), using {num_gpus} GPU(s) for evaluation")
        if num_gpus > 1:
            print("Using DataParallel for multi-GPU evaluation")
    else:
        device = torch.device('cpu')
        num_gpus = 0
        print("Using CPU for evaluation")
    
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
        d_model=config['d_model'],
        N=config['N'],
        h=config['h'],
        dropout=config['dropout'],
        d_ff=config['d_ff'],
        positional_encoding=config.get('positional_encoding', 'rope')
    )
    
    # Wrap model with DataParallel if multiple GPUs are available
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"Model wrapped with DataParallel using {num_gpus} GPUs")
    
    model = model.to(device)

    # Load the pretrained weights using auto-detection
    model_filename = get_checkpoint_path_for_epoch(config, epoch_number)
    if model_filename:
        print(f"Loading model weights from epoch {epoch_number}: {model_filename}")
        try:
            state = torch.load(model_filename, map_location=device, weights_only=False)
            # Handle DataParallel checkpoint loading
            if num_gpus > 1:
                # If checkpoint was saved with DataParallel, load it directly
                if 'module.' in list(state['model_state_dict'].keys())[0]:
                    model.load_state_dict(state['model_state_dict'])
                else:
                    # If checkpoint was saved without DataParallel, add 'module.' prefix
                    new_state_dict = {}
                    for k, v in state['model_state_dict'].items():
                        new_state_dict['module.' + k] = v
                    model.load_state_dict(new_state_dict)
            else:
                # Single GPU loading
                if 'module.' in list(state['model_state_dict'].keys())[0]:
                    # Remove 'module.' prefix for single GPU
                    new_state_dict = {}
                    for k, v in state['model_state_dict'].items():
                        new_state_dict[k.replace('module.', '')] = v
                    model.load_state_dict(new_state_dict)
                else:
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

    # Get test dataset and true number of examples
    test_dataloader, tokenizer_src, tokenizer_tgt, num_test_examples = get_test_dataset(config)

    print(f"Evaluating model on {num_test_examples} test examples...")

    # Warn if batch size is too small for multi-GPU
    if num_gpus > 1 and config['batch_size'] < num_gpus:
        print(f"Warning: Batch size ({config['batch_size']}) is less than number of GPUs ({num_gpus}). Increase batch size for better GPU utilization.")

    model.eval()
    predicted_greedy = []
    predicted_beam = []
    predicted_top_k = []
    expected = []
    
    # Progress tracking and optional subset limiting
    total_batches = len(test_dataloader)
    processed_samples = 0
    max_test_samples = config.get('max_test_samples', None)

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
            encoder_input = batch['encoder_input'].to(device)  # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)    # (batch_size, 1, seq_len, seq_len)
            expected_texts = batch['label_text']  # List of expected texts

            batch_size = encoder_input.size(0)
            
            # Initialize output lists for this batch
            outputs_greedy = []
            outputs_beam = []
            outputs_top_k = []
            
            # OPTIMIZATION: Run encoder once per batch instead of per sample
            model_module = get_model_module(model)
            encoder_output = model_module.encode(encoder_input, encoder_mask)  # (batch_size, seq_len, d_model)
            
            # OPTIMIZATION: Use batch greedy decoding for maximum efficiency
            if config.get('strategy', 'all') in ['greedy', 'all']:
                if config.get('use_batch_greedy', True):
                    try:
                        batch_greedy_outputs = batch_greedy_decode(
                            model_module, encoder_output, encoder_mask, tokenizer_tgt, 
                            config['seq_len'], device, 
                            repetition_penalty=config.get('repetition_penalty', 1.0),
                            unk_penalty=config.get('unk_penalty', 1.0)
                        )
                        # Split batch outputs into individual samples
                        for i in range(batch_size):
                            outputs_greedy.append(batch_greedy_outputs[i])
                    except Exception as e:
                        print(f"Warning: Batch greedy decoding failed: {e}")
                        # Fallback to individual decoding
                        for i in range(batch_size):
                            try:
                                single_encoder_output = encoder_output[i:i+1]
                                single_encoder_mask = encoder_mask[i:i+1]
                                greedy_out = greedy_decode_optimized(
                                    model_module, single_encoder_output, single_encoder_mask, tokenizer_tgt, 
                                    config['seq_len'], device, 
                                    repetition_penalty=config.get('repetition_penalty', 1.0),
                                    unk_penalty=config.get('unk_penalty', 1.0)
                                )
                                outputs_greedy.append(greedy_out)
                            except Exception as e2:
                                print(f"Warning: Individual greedy decoding failed for sample {i}: {e2}")
                                outputs_greedy.append(torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], device=device))
                else:
                    # Use individual optimized decoding
                    for i in range(batch_size):
                        try:
                            single_encoder_output = encoder_output[i:i+1]
                            single_encoder_mask = encoder_mask[i:i+1]
                            greedy_out = greedy_decode_optimized(
                                model_module, single_encoder_output, single_encoder_mask, tokenizer_tgt, 
                                config['seq_len'], device, 
                                repetition_penalty=config.get('repetition_penalty', 1.0),
                                unk_penalty=config.get('unk_penalty', 1.0)
                            )
                            outputs_greedy.append(greedy_out)
                        except Exception as e:
                            print(f"Warning: Individual greedy decoding failed for sample {i}: {e}")
                            outputs_greedy.append(torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], device=device))
            
            # For beam search and top-k, we still need individual decoding due to their complexity
            if config.get('strategy', 'all') in ['beam', 'all']:
                for i in range(batch_size):
                    try:
                        single_encoder_output = encoder_output[i:i+1]
                        single_encoder_mask = encoder_mask[i:i+1]
                        beam_out = beam_search_decode_optimized(
                            model_module, single_encoder_output, single_encoder_mask, tokenizer_tgt, 
                            config['seq_len'], device, config['beam_size'],
                            repetition_penalty=config.get('repetition_penalty', 1.0),
                            unk_penalty=config.get('unk_penalty', 1.0),
                            length_penalty=config.get('length_penalty', 1.0)
                        )
                        outputs_beam.append(beam_out)
                    except Exception as e:
                        print(f"Warning: Beam search failed for sample {i}: {e}")
                        outputs_beam.append(torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], device=device))
            
            if config.get('strategy', 'all') in ['topk', 'all']:
                for i in range(batch_size):
                    try:
                        single_encoder_output = encoder_output[i:i+1]
                        single_encoder_mask = encoder_mask[i:i+1]
                        topk_out = top_k_sampling_decode_optimized(
                            model_module, single_encoder_output, single_encoder_mask, tokenizer_tgt, 
                            config['seq_len'], device, config['top_k'],
                            repetition_penalty=config.get('repetition_penalty', 1.0),
                            unk_penalty=config.get('unk_penalty', 1.0),
                            temperature=config.get('temperature', 1.0)
                        )
                        outputs_top_k.append(topk_out)
                    except Exception as e:
                        print(f"Warning: Top-k sampling failed for sample {i}: {e}")
                        outputs_top_k.append(torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], device=device))

            tgt_texts = batch['tgt_text']
            expected.extend(tgt_texts)

            # Decode all predictions and extend lists
            if outputs_greedy:
                model_output_texts_greedy = [tokenizer_tgt.decode(output.detach().cpu().numpy()) for output in outputs_greedy]
                predicted_greedy.extend(model_output_texts_greedy)
            if outputs_beam:
                model_output_texts_beam = [tokenizer_tgt.decode(output.detach().cpu().numpy()) for output in outputs_beam]
                predicted_beam.extend(model_output_texts_beam)
            if outputs_top_k:
                model_output_texts_top_k = [tokenizer_tgt.decode(output.detach().cpu().numpy()) for output in outputs_top_k]
                predicted_top_k.extend(model_output_texts_top_k)
            
            # Update progress and optionally stop early
            processed_samples += batch_size
            if processed_samples % 100 == 0:
                print(f"Processed {processed_samples}/{num_test_examples} samples...")
            if (max_test_samples is not None) and (processed_samples >= max_test_samples):
                print(f"Stopping early after {processed_samples} samples (--max_test_samples).")
                break

    # Save predictions and expected translations to files
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_filepath_greedy = output_dir / f"predictions_greedy_epoch_{epoch_number}.txt"
    pred_filepath_beam = output_dir / f"predictions_beam_epoch_{epoch_number}.txt"
    pred_filepath_top_k = output_dir / f"predictions_top_k_epoch_{epoch_number}.txt"
    exp_filepath = output_dir / f"expected_epoch_{epoch_number}.txt"

    if predicted_greedy:
        with open(pred_filepath_greedy, "w", encoding="utf-8") as f:
            f.write("\n".join(predicted_greedy))
    if predicted_beam:
        with open(pred_filepath_beam, "w", encoding="utf-8") as f:
            f.write("\n".join(predicted_beam))
    if predicted_top_k:
        with open(pred_filepath_top_k, "w", encoding="utf-8") as f:
            f.write("\n".join(predicted_top_k))
    
    with open(exp_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(expected))
        
    print(f"Predictions saved to:")
    if predicted_greedy:
        print(f"  Greedy: {pred_filepath_greedy}")
    if predicted_beam:
        print(f"  Beam:   {pred_filepath_beam}")
    if predicted_top_k:
        print(f"  Top-k:  {pred_filepath_top_k}")
    print(f"Expected: {exp_filepath}")

    print("=" * 50)
    print(f"EVALUATION RESULTS FOR EPOCH {epoch_number}")
    print("=" * 50)

    if metrics_available:
        # Calculate BLEU and BERTScore for selected decoding methods
        if predicted_greedy:
            print("\n--- GREEDY DECODING ---")
            bleu_greedy = bleu_metric([p.lower() for p in predicted_greedy], [[e.lower()] for e in expected])
            print(f"BLEU Score:      {bleu_greedy:.4f}")
            P, R, F1 = bert_scorer.score(predicted_greedy, expected)
            bert_f1_greedy = F1.mean()
            print(f"BERTScore (F1):  {bert_f1_greedy:.4f}")
        if predicted_beam:
            print("\n--- BEAM SEARCH DECODING ---")
            bleu_beam = bleu_metric([p.lower() for p in predicted_beam], [[e.lower()] for e in expected])
            print(f"BLEU Score:      {bleu_beam:.4f}")
            P, R, F1 = bert_scorer.score(predicted_beam, expected)
            bert_f1_beam = F1.mean()
            print(f"BERTScore (F1):  {bert_f1_beam:.4f}")
        if predicted_top_k:
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
    # Check for GPU availability and count
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        requested_gpus = config.get('num_gpus', available_gpus)
        if requested_gpus is None:
            requested_gpus = available_gpus
        num_gpus = min(requested_gpus, available_gpus)
        device = torch.device('cuda')
        print(f"Found {available_gpus} GPU(s), using {num_gpus} GPU(s) for translation")
        if num_gpus > 1:
            print("Using DataParallel for multi-GPU translation")
    else:
        device = torch.device('cpu')
        num_gpus = 0
        print("Using CPU for translation")
    
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
        d_model=config['d_model'],
        N=config['N'],
        h=config['h'],
        dropout=config['dropout'],
        d_ff=config['d_ff'],
        positional_encoding=config.get('positional_encoding', 'rope')
    )
    
    # Wrap model with DataParallel if multiple GPUs are available
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"Model wrapped with DataParallel using {num_gpus} GPUs")
    
    model = model.to(device)

    # Load the pretrained weights using auto-detection
    model_filename = get_checkpoint_path_for_epoch(config, epoch_number)
    if model_filename:
        print(f"Loading model weights from epoch {epoch_number}: {model_filename}")
        try:
            state = torch.load(model_filename, map_location=device, weights_only=False)
            # Handle DataParallel checkpoint loading
            if num_gpus > 1:
                # If checkpoint was saved with DataParallel, load it directly
                if 'module.' in list(state['model_state_dict'].keys())[0]:
                    model.load_state_dict(state['model_state_dict'])
                else:
                    # If checkpoint was saved without DataParallel, add 'module.' prefix
                    new_state_dict = {}
                    for k, v in state['model_state_dict'].items():
                        new_state_dict['module.' + k] = v
                    model.load_state_dict(new_state_dict)
            else:
                # Single GPU loading
                if 'module.' in list(state['model_state_dict'].keys())[0]:
                    # Remove 'module.' prefix for single GPU
                    new_state_dict = {}
                    for k, v in state['model_state_dict'].items():
                        new_state_dict[k.replace('module.', '')] = v
                    model.load_state_dict(new_state_dict)
                else:
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
        model_out_greedy = greedy_decode(
            model, source, source_mask, tokenizer_src, tokenizer_tgt, seq_len, device,
            repetition_penalty=config.get('repetition_penalty', 1.0),
            unk_penalty=config.get('unk_penalty', 1.0)
        )
        output_text_greedy = tokenizer_tgt.decode(model_out_greedy.tolist())

        # Beam Search Decode
        model_out_beam = beam_search_decode(
            model, source, source_mask, tokenizer_src, tokenizer_tgt, seq_len, device, config['beam_size'],
            repetition_penalty=config.get('repetition_penalty', 1.0),
            unk_penalty=config.get('unk_penalty', 1.0),
            length_penalty=config.get('length_penalty', 1.0)
        )
        output_text_beam = tokenizer_tgt.decode(model_out_beam.tolist())

        # Top-k Sampling
        model_out_top_k = top_k_sampling_decode(
            model, source, source_mask, tokenizer_src, tokenizer_tgt, seq_len, device, config['top_k'],
            repetition_penalty=config.get('repetition_penalty', 1.0),
            unk_penalty=config.get('unk_penalty', 1.0),
            temperature=config.get('temperature', 1.0)
        )
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
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use (default: all available)")
    parser.add_argument("--strategy", choices=["greedy", "beam", "topk", "all"], default="all", help="Decoding strategy for translation/evaluation")
    parser.add_argument("--positional_encoding", type=str, default="rope", choices=["rope", "relative"], help="Positional encoding type")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Limit evaluation to first N samples (for quick runs)")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k for top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Penalty for repeating tokens")
    parser.add_argument("--unk_penalty", type=float, default=1.5, help="Penalty for UNK tokens")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty for beam search")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for top-k sampling")
    parser.add_argument("--use_batch_greedy", action="store_true", default=True, help="Use optimized batch greedy decoding")
    
    args = parser.parse_args()
    
    config = get_config()
    config['batch_size'] = args.batch_size
    config['num_gpus'] = args.gpus
    config['strategy'] = args.strategy
    config['positional_encoding'] = args.positional_encoding
    config['num_workers'] = args.num_workers
    config['max_test_samples'] = args.max_test_samples
    config['beam_size'] = args.beam_size
    config['top_k'] = args.top_k
    config['repetition_penalty'] = args.repetition_penalty
    config['unk_penalty'] = args.unk_penalty
    config['length_penalty'] = args.length_penalty
    config['temperature'] = args.temperature
    config['use_batch_greedy'] = args.use_batch_greedy
    
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
