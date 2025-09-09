import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
import torchmetrics

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


def run_evaluation(model, dataset, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, name, num_examples=2 ):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in dataset:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

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

    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar(f'{name} cer', cer, global_step)
        writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar(f'{name} wer', wer, global_step)
        writer.flush()

        metric = torchmetrics.BLEUScore()
        bleu = metric([p.lower() for p in predicted], [[e.lower()] for e in expected])
        writer.add_scalar(f'{name} BLEU', bleu, global_step)
        writer.flush()

def get_dataset(config):
    # Load the pre-split datasets
    train_data = load_dataset_by_split(config, 'train')
    val_data = load_dataset_by_split(config, 'val')
    test_data = load_dataset_by_split(config, 'test')

    tokenizer_src = get_or_build_tokenizer(config, None, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, None, config['lang_tgt'])

    # Create dataset objects
    train_dataset = BilingualDataset(train_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    validation_dataset = BilingualDataset(val_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_dataset = BilingualDataset(test_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)  # No shuffling for validation
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # No shuffling for test

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
        d_ff=config['d_ff']
    )
    return model

def train_model(config):
    # Check for GPU availability and count
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        requested_gpus = config.get('num_gpus', available_gpus)
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
    
    model = model.to(device)
    
    writer = SummaryWriter(config['experiment_name'])
    
    # Adjust learning rate for multi-GPU training
    effective_lr = config['lr'] * num_gpus if num_gpus > 1 else config['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr, eps=1e-9)
    print(f"Effective learning rate: {effective_lr}")

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

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            model_module = get_model_module(model)
            encoder_output = model_module.encode(encoder_input, encoder_mask)
            decoder_output = model_module.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model_module.project(decoder_output)

            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Update learning rate
            d_model = config['d_model']
            warmup_steps = config['warmup_steps']
            lr = (d_model ** -0.5) * min((global_step + 1) ** -0.5, (global_step + 1) * (warmup_steps ** -1.5))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            writer.add_scalar('learning_rate', lr, global_step)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
        # Save model checkpoint
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
        }, model_filename)

    # Final evaluation on the test set
    run_evaluation(model, test_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, print, 0, writer, "test")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the transformer model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=12, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=10**-4, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=150, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use (default: all available)")
    
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
    
    train_model(config)
