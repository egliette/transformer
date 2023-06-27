import math
import time
import os

import pandas as pd
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.optim import Adam

import utils.data_utils as data_utils
import utils.model_utils as model_utils
from utils.bleu import idx_to_word, get_bleu
from utils.utils import epoch_time, create_dir
from models.model.transformer import Transformer


def train(model, iterator, optimizer, criterion, clip, batch_id):
    model.train()
    device = model.device
    epoch_loss = 0
    with tqdm(enumerate(iterator), total=len(iterator)) as pbar:
        pbar.set_description(f"Epoch {batch_id}")
        for i, batch in pbar:
            src = batch["src"].to(device)
            trg = batch["tgt"].to(device)

            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            break

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, parallel_vocab):
    model.eval()
    device = model.device
    epoch_loss = 0
    batch_bleu = list()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch["src"].to(device)
            trg = batch["tgt"].to(device)
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()
            total_bleu = list()
            for j in range(batch["tgt"].shape[0]):
                  trg_words = idx_to_word(batch["tgt"].to(device)[j], parallel_vocab.tgt)
                  output_words = output[j].max(dim=1)[1]
                  output_words = idx_to_word(output_words, parallel_vocab.tgt)
                  bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                  total_bleu.append(bleu)

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu



print("Load config file...")
config = data_utils.get_config("config.yml")

for key, value in config.items():
    globals()[key] = value

print("Ensure directory and load file paths...")
create_dir(checkpoint["dir"])
checkpoint_fpath = "/".join([checkpoint["dir"], checkpoint["name"]])
results_fpath = "/".join([checkpoint["dir"], checkpoint["results"]])
vocab_fpath = "/".join([checkpoint["dir"], checkpoint["parallel_vocab"]])
dataloaders_fpath = "/".join([checkpoint["dir"], checkpoint["dataloaders"]])

print("Load prepared dataloaders and vocabulary...")
envi_vocab = torch.load(vocab_fpath)
dataloaders = torch.load(dataloaders_fpath)
train_loader = dataloaders["train_loader"]
valid_loader = dataloaders["valid_loader"]

print("Load model & optimizer & criterion...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc_voc_size = len(envi_vocab.src)
dec_voc_size = len(envi_vocab.tgt)

model = Transformer(src_pad_idx=envi_vocab.src.pad_id,
                    trg_pad_idx=envi_vocab.tgt.pad_id,
                    trg_sos_idx=envi_vocab.tgt.bos_id,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

optimizer = Adam(params=model.parameters(),
                lr=init_lr,
                weight_decay=weight_decay,
                eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                verbose=True,
                                                factor=factor,
                                                patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=envi_vocab.src.pad_id)

# Load checkpoint
begin_epoch = 1
best_loss = float("inf")
if os.path.isfile(checkpoint_fpath):
  checkpoint_dict = torch.load(checkpoint_fpath)
  best_loss = checkpoint_dict["loss"]
  begin_epoch = checkpoint_dict["epoch"] + 1
  model.load_state_dict(checkpoint_dict["model_state_dict"])
  optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
  if begin_epoch <= total_epoch:
    print(f"Continue from epoch {begin_epoch}...")
else:
  model.apply(model_utils.initialize_weights)

print(f"The model has {model_utils.count_parameters(model):,} trainable parameters")

columns = ["epoch", "train_loss", "valid_loss", "valid_BLEU"]
results_df = pd.DataFrame(columns=columns)
if os.path.isfile(results_fpath):
  results_df = pd.read_csv(results_fpath)

print(f"Start training & evaluating...")
for epoch in range(begin_epoch, total_epoch+1):
    start_time = time.time()
    train_loss = train(model, train_loader, optimizer, criterion, clip, epoch)
    valid_loss, valid_BLEU = evaluate(model, valid_loader, criterion, envi_vocab)
    end_time = time.time()

    if epoch > warmup:
        scheduler.epoch(valid_loss)

    results_df.loc[len(results_df), columns] = epoch, train_loss, valid_loss, valid_BLEU
    results_df.to_csv(results_fpath, index=False)

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save({"epoch": epoch,
                    "loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()},
                    checkpoint_fpath)


    print(f'Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
    print(f'\tBLEU Score: {valid_BLEU:.3f}')

print("Finish training!")