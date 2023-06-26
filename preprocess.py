import torch
from torch.utils.data import DataLoader

import utils.data_utils as data_utils
from dataset import ParallelDataset
from tokenizer import EnTokenizer, ViTokenizer
from vocabulary import Vocabulary, ParallelVocabulary


# Load config file
config = data_utils.get_config("config.yml")

for key, value in config.items():
    globals()[key] = value


# Load tokenizers
en_tok = EnTokenizer()
en_vocab = Vocabulary(en_tok)
en_vocab.add_words_from_corpus(corpus_fpath=path["src"]["train"])

vi_tok = ViTokenizer()
vi_vocab = Vocabulary(vi_tok)
vi_vocab.add_words_from_corpus(corpus_fpath=path["tgt"]["train"])

envi_vocab = ParallelVocabulary(en_vocab, vi_vocab)


# Load datasets
train_set = ParallelDataset(path["src"]["train"], path["tgt"]["train"],
                            parallel_vocab=envi_vocab)
valid_set = ParallelDataset(path["src"]["valid"], path["tgt"]["valid"],
                            parallel_vocab=envi_vocab)
test_set = ParallelDataset(path["src"]["test"], path["tgt"]["test"],
                           parallel_vocab=envi_vocab)

# Load dataloaders
train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=envi_vocab.collate_fn)
valid_loader = DataLoader(valid_set,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=envi_vocab.collate_fn)
test_loader = DataLoader(test_set,
                         batch_size=batch_size,
                         shuffle=False,
                         collate_fn=envi_vocab.collate_fn)

dataloaders = {"train_loader": train_loader,
               "valid_loader": valid_loader,
               "test_loader": test_loader,}

torch.save(envi_vocab, path["parallel_vocab"])
torch.save(dataloaders, path["dataloaders"])