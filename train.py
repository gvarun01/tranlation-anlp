import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
import torchmetrics
import time

from model import build_transformer
from utils import (
    get_config,
    get_weights_path,
    find_latest_checkpoint,
    get_or_build_tokenizer,
    BilingualDataset,
    greedy_decode,
    load_dataset_by_split,
    get_model_module,
)


def run_evaluation(model, dataset, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, name, num_examples=2, loss_fn=None):
    model.eval()
    count = 0
    total_loss = 0.0
    num_batches = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in dataset:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # Compute loss if loss_fn provided
            if loss_fn is not None:
                model_module = get_model_module(model)
                encoder_output = model_module.encode(encoder_input, encoder_mask)
                decoder_output = model_module.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model_module.project(decoder_output)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                total_loss += loss.item()
                num_batches += 1

            # Greedy decode for metrics
            assert encoder_input.size(0) == 1, "Batch size should be 1 for evaluation"
            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]
            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            source_texts.append(src_text)
            expected.append(tgt_text)
            predicted.append(model_output_text)

            if count <= num_examples:
                print_msg('-'*console_width)
                print_msg(f"--- {name.upper()} EXAMPLE {count} ---")
                print_msg(f"{f'SOURCE: ':>12}{src_text}")
                print_msg(f"{f'TARGET: ':>12}{tgt_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_output_text}")

    # Compute metrics
    cer = None
    wer = None
    bleu = None
    val_loss = None
    
    if len(predicted) > 0:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        if writer:
            writer.add_scalar(f'{name} cer', cer, global_step)
            writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        if writer:
            writer.add_scalar(f'{name} wer', wer, global_step)
            writer.flush()

        metric = torchmetrics.BLEUScore()
        bleu = metric([p.lower() for p in predicted], [[e.lower()] for e in expected])
        if writer:
            writer.add_scalar(f'{name} BLEU', bleu, global_step)
            writer.flush()

    if num_batches > 0:
        val_loss = total_loss / num_batches
        if writer:
            writer.add_scalar(f'{name} loss', val_loss, global_step)
            writer.flush()

    return {"bleu": float(bleu) if bleu is not None else None,
            "wer": float(wer) if wer is not None else None,
            "cer": float(cer) if cer is not None else None,
            "loss": val_loss}

def get_dataset(config):
    # Load the pre-split datasets
    train_data = load_dataset_by_split(config, 'train')
    val_data = load_dataset_by_split(config, 'val')
    test_data = load_dataset_by_split(config, 'test')

    tokenizer_src = get_or_build_tokenizer(config, None, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, None, config['lang_tgt'])

    # Create dataset objects
    train_dataset = BilingualDataset(train_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'], device=None)  # CPU for data loading
    validation_dataset = BilingualDataset(val_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'], device=None)  # CPU for data loading
    test_dataset = BilingualDataset(test_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'], device=None)  # CPU for data loading

    # Create data loaders with optimized settings for multi-GPU
    num_workers = min(4, config.get('num_workers', 4))  # Limit workers for Kaggle
    pin_memory = torch.cuda.is_available()  # Pin memory for faster GPU transfer
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=0,  # No workers for validation to avoid issues
        pin_memory=pin_memory
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=0,  # No workers for test to avoid issues
        pin_memory=pin_memory
    )

    print(f"Dataset loaded with pre-defined splits:")
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Validation: {len(validation_dataset)} examples") 
    print(f"  Test: {len(test_dataset)} examples")

    return train_dataloader, validation_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model'],
        N=config['N'],
        h=config['h'],
        dropout=config['dropout'],
        d_ff=config['d_ff'],
        positional_encoding=config.get('positional_encoding', 'rope')
    )
    return model

def train_model(config):
    # Check for GPU availability and count
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        requested_gpus = config.get('num_gpus', available_gpus)
        if requested_gpus is None:
            requested_gpus = available_gpus
        num_gpus = min(requested_gpus, available_gpus)
        device = torch.device('cuda')
        print(f"Found {available_gpus} GPU(s), using {num_gpus} GPU(s)")
        if num_gpus > 1:
            print("Using DataParallel for multi-GPU training")
    else:
        device = torch.device('cpu')
        num_gpus = 0
        print("Using CPU for training")

    # Set up the base path for saving models and tokenizers
    base_path = Path('.')
    
    # Make sure the model folder exists
    (base_path / config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, validation_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    
    # Wrap model with DataParallel if multiple GPUs are available
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"Model wrapped with DataParallel using {num_gpus} GPUs")
        
        # Debug multi-GPU setup
        print(f"DataParallel device_ids: {list(range(num_gpus))}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Available GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
        print(f"CUDA memory per GPU: {[torch.cuda.get_device_properties(i).total_memory // 1024**3 for i in range(torch.cuda.device_count())]} GB")
    
    model = model.to(device)
    
    writer = SummaryWriter(config['experiment_name'])
    
    # Adjust learning rate for multi-GPU training
    effective_lr = config['lr'] * num_gpus if num_gpus > 1 else config['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr, betas=(0.9, 0.98), eps=1e-9)
    print(f"Effective learning rate: {effective_lr}")
    print(f"DataLoader workers: {num_workers}, Pin memory: {pin_memory}")

    initial_epoch = 0
    global_step = 0
    
    # Auto-detect and load the latest checkpoint
    checkpoint_path, latest_epoch = find_latest_checkpoint(config)
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            # Handle DataParallel checkpoint loading
            if num_gpus > 1:
                # If checkpoint was saved with DataParallel, load it directly
                if 'module.' in list(state['model_state_dict'].keys())[0]:
                    model.load_state_dict(state['model_state_dict'])
                else:
                    # If checkpoint was saved without DataParallel, remove 'module.' prefix
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
            
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state.get('global_step', 0)
            print(f"Resumed training from epoch {initial_epoch} (continuing from epoch {latest_epoch})")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            initial_epoch = 0
            global_step = 0
    else:
        print("Starting training from scratch...")
        initial_epoch = 0

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.get('use_amp', False) and torch.cuda.is_available() else None
    
    # Compile model for speed if enabled
    if config.get('compile_model', False) and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Early stopping and best model tracking
    best_val_bleu = None
    best_val_loss = None
    best_model_path = None
    patience_counter = 0
    early_stopping_patience = config.get('early_stopping_patience', 5)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    print(f"Training with gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Mixed precision: {scaler is not None}")
    
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        epoch_start_time = time.time()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        
        for batch_idx, batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # Mixed precision forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    model_module = get_model_module(model)
                    encoder_output = model_module.encode(encoder_input, encoder_mask)
                    decoder_output = model_module.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                    proj_output = model_module.project(decoder_output)
                    loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                    loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
            else:
                model_module = get_model_module(model)
                encoder_output = model_module.encode(encoder_input, encoder_mask)
                decoder_output = model_module.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model_module.project(decoder_output)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation

            batch_iterator.set_postfix({f"loss": f"{loss.item() * gradient_accumulation_steps:6.3f}"})

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update learning rate
            d_model = config['d_model']
            warmup_steps = config['warmup_steps']
            lr = (d_model ** -0.5) * min((global_step + 1) ** -0.5, (global_step + 1) * (warmup_steps ** -1.5))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            writer.add_scalar('learning_rate', lr, global_step)
            writer.add_scalar('train_loss', loss.item() * gradient_accumulation_steps, global_step)

            # Gradient accumulation: only step every N batches
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping for stability
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        
        # Validation evaluation at epoch end (only if frequency allows)
        val_eval_frequency = config.get('val_eval_frequency', 1)
        if epoch % val_eval_frequency == 0 or epoch == config['num_epochs'] - 1:
            val_metrics = run_evaluation(model, validation_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda *args, **kwargs: None, global_step, writer, "val", num_examples=0, loss_fn=loss_fn)
            current_bleu = val_metrics.get('bleu') if val_metrics else None
            current_loss = val_metrics.get('loss') if val_metrics else None
            
            # Track best model by BLEU (primary) and loss (secondary)
            improved = False
            if current_bleu is not None:
                if best_val_bleu is None or current_bleu > best_val_bleu:
                    best_val_bleu = current_bleu
                    improved = True
                    print(f"New best BLEU: {best_val_bleu:.4f} at epoch {epoch:02d}")
                elif current_loss is not None and best_val_loss is not None and current_loss < best_val_loss and current_bleu >= best_val_bleu * 0.99:  # Within 1% of best BLEU
                    best_val_loss = current_loss
                    improved = True
                    print(f"New best loss: {best_val_loss:.4f} at epoch {epoch:02d}")
            
            if current_loss is not None:
                if best_val_loss is None:
                    best_val_loss = current_loss
                elif current_loss < best_val_loss:
                    best_val_loss = current_loss
            
            # Early stopping check
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs (patience: {early_stopping_patience})")
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch:02d}")
                    break

        # Save model checkpoint (only if frequency allows)
        save_frequency = config.get('save_frequency', 1)
        if epoch % save_frequency == 0 or epoch == config['num_epochs'] - 1:
            model_filename = get_weights_path(config, f'{epoch:02d}')
            
            # Handle DataParallel checkpoint saving
            if num_gpus > 1:
                # Save the underlying model state_dict (without DataParallel wrapper)
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
                
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "val_bleu": current_bleu,
                "val_loss": current_loss,
            }, model_filename)

            # Save best model if improved
            if improved and current_bleu is not None:
                best_model_path = get_weights_path(config, 'best')
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "val_bleu": best_val_bleu,
                    "val_loss": best_val_loss,
                }, best_model_path)
                print(f"Best model saved to {best_model_path}")

        # Log epoch timing
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch:02d} completed in {epoch_time:.2f}s")

    # Final evaluation on the test set
    print("Running final test evaluation...")
    run_evaluation(model, test_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, print, 0, writer, "test", loss_fn=loss_fn)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the transformer model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=12, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=10**-4, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=150, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use (default: all available)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--compile_model", action="store_true", help="Use torch.compile for speed")
    parser.add_argument("--val_eval_frequency", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--save_frequency", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--positional_encoding", type=str, default="rope", choices=["rope", "relative"], help="Positional encoding type")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    args = parser.parse_args()
    
    warnings.filterwarnings("ignore")
    config = get_config()
    
    # Override config with command line arguments
    config['batch_size'] = args.batch_size
    config['num_epochs'] = args.num_epochs
    config['lr'] = args.lr
    config['seq_len'] = args.seq_len
    config['d_model'] = args.d_model
    config['num_gpus'] = args.gpus
    config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    config['early_stopping_patience'] = args.early_stopping_patience
    config['use_amp'] = args.use_amp
    config['compile_model'] = args.compile_model
    config['val_eval_frequency'] = args.val_eval_frequency
    config['save_frequency'] = args.save_frequency
    config['positional_encoding'] = args.positional_encoding
    config['num_workers'] = args.num_workers
    
    train_model(config)
