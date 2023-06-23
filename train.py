import math
import time

from tqdm import tqdm
import torch
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader

import data_utils
import model_utils
from dataset import ParallelDataset
from vocabulary import Vocabulary, ParallelVocabulary
from tokenizer import EnTokenizer, ViTokenizer
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time



# Load config file
config = data_utils.get_config("config.yml")

for key, value in config.items():
    globals()[key] = value


# Load datasets
train_set = ParallelDataset(path["src"]["train"],
                            path["tgt"]["train"])
valid_set = ParallelDataset(path["src"]["valid"],
                            path["tgt"]["valid"])
test_set = ParallelDataset(path["src"]["test"],
                           path["tgt"]["test"])

corpus = train_set


# Load tokenizers
en_tok = EnTokenizer()
en_vocab = Vocabulary(en_tok)
en_corpus = [src for (src, tgt) in corpus]
en_vocab.add_words_from_corpus(en_corpus)

vi_tok = ViTokenizer()
vi_vocab = Vocabulary(vi_tok)
vi_corpus = [tgt for (src, tgt) in corpus]
vi_vocab.add_words_from_corpus(vi_corpus)

envi_vocab = ParallelVocabulary(en_vocab, vi_vocab)


# Load dataloaders
train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=lambda example: data_utils.collate_fn(envi_vocab, example))
valid_loader = DataLoader(valid_set,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=lambda example: data_utils.collate_fn(envi_vocab, example))
test_loader = DataLoader(test_set,
                          batch_size=batch_size,
                          shuffle=False,
                          collate_fn=lambda example: data_utils.collate_fn(envi_vocab, example))


# Load model
inf = float("inf")
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


print(f'The model has {model_utils.count_parameters(model):,} trainable parameters')
model.apply(model_utils.initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=envi_vocab.src.pad_id)


def train(model, iterator, optimizer, criterion, clip, batch_id):
    model.train()
    epoch_loss = 0

    with tqdm(enumerate(iterator), total=len(iterator)) as pbar:
        pbar.set_description(f"Epoch {batch_id}: ")
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

    return epoch_loss / len(iterator)


def idx_to_word_new(x, vocab):
    words = []
    for i in x:
        word = vocab.id2word[i.item()]
        if '<' not in word:
            words.append(word)
    words = " ".join(words)
    return words


def evaluate(model, iterator, criterion, parallel_vocab):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch["src"].to(device)
            trg = batch["tgt"].to(device)
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()
            total_bleu = []
            for j in range(len(batch)):
                  trg_words = idx_to_word(batch["tgt"].to(device)[j], parallel_vocab.tgt)
                  output_words = output[j].max(dim=1)[1]
                  output_words = idx_to_word(output_words, parallel_vocab.tgt)
                  bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                  total_bleu.append(bleu)


            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, clip, step+1)
        valid_loss, bleu = evaluate(model, valid_loader, criterion, envi_vocab)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)