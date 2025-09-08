import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from datasets import load_dataset
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
import torchmetrics
from datasets import load_dataset

from model import build_transformer
from utils import (
    get_config,
    get_weights_path,
    get_or_build_tokenizer,
    BilingualDataset,
    greedy_decode,
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
    # Load the dataset from local files
    src_lang = config['lang_src']
    tgt_lang = config['lang_tgt']
    data_path = Path('data')
    src_path = data_path / f"EUbookshop.{src_lang}"
    tgt_path = data_path / f"EUbookshop.{tgt_lang}"

    with open(src_path, 'r', encoding='utf-8') as f:
        src_lines = f.read().splitlines()
    with open(tgt_path, 'r', encoding='utf-8') as f:
        tgt_lines = f.read().splitlines()

    # Create a dataset in the expected format
    dataset_raw = []
    for src, tgt in zip(src_lines, tgt_lines):
        dataset_raw.append({
            'translation': {
                src_lang: src,
                tgt_lang: tgt
            }
        })

    tokenizer_src = get_or_build_tokenizer(config, None, src_lang)
    tokenizer_tgt = get_or_build_tokenizer(config, None, tgt_lang)

    # Filter and split the dataset
    filtered_data = [
        item for item in dataset_raw
        if len(tokenizer_src.encode(item['translation'][config['lang_src']]).ids) <= config['seq_len'] - 2
        and len(tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids) <= config['seq_len'] - 1
    ]
    
    # Split the dataset into train, validation, and test sets
    train_ds_size = int(0.8 * len(filtered_data))
    val_ds_size = int(0.1 * len(filtered_data))
    test_ds_size = len(filtered_data) - train_ds_size - val_ds_size
    train_ds_raw, val_ds_raw, test_ds_raw = random_split(filtered_data, [train_ds_size, val_ds_size, test_ds_size])

    train_dataset = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    validation_dataset = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_dataset = BilingualDataset(test_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model'],
    )
    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up the base path for saving models and tokenizers
    base_path = Path('.')
    gdrive_path = config.get('gdrive_path')
    if gdrive_path:
        base_path = Path(gdrive_path)
    
    # Make sure the model folder exists
    (base_path / config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, validation_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    
    # Check for existing checkpoints to resume from
    if config['preload']:
        model_filename = get_weights_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        # Auto-resume from latest checkpoint if available
        base_path = Path('.')
        gdrive_path = config.get('gdrive_path')
        if gdrive_path:
            base_path = Path(gdrive_path)
        
        model_folder = base_path / config['model_folder']
        if model_folder.exists():
            # Find the latest checkpoint
            checkpoints = list(model_folder.glob(f"{config['model_basename']}*.pt"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
                print(f"Found existing checkpoint: {latest_checkpoint}")
                print("Do you want to resume from this checkpoint? (y/n)")
                response = input().strip().lower()
                if response == 'y':
                    state = torch.load(latest_checkpoint)
                    model.load_state_dict(state['model_state_dict'])
                    initial_epoch = state['epoch'] + 1
                    optimizer.load_state_dict(state['optimizer_state_dict'])
                    global_step = state['global_step']
                    print(f"Resumed training from epoch {initial_epoch}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

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
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        }, model_filename)

    # Final evaluation on the test set
    run_evaluation(model, test_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, print, 0, writer, "test")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
