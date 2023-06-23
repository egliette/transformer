import torch
from torch.utils.data import Dataset

class ParallelDataset(Dataset):
    """Load dataset from source and target text files"""

    def __init__(self, src_fpath, tgt_fpath, parallel_vocab=None, tokenized_fpath=None):
        self.src_sents = self._read_data(src_fpath)
        self.tgt_sents = self._read_data(tgt_fpath)
        # self.parallel_vocab = parallel_vocab

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, id):
        if isinstance(id, int):
            # return self.tokenize_pair(id)
            return self.src_sents[id], self.tgt_sents[id]
        elif isinstance(id, slice):
            start = 0 if id.start == None else id.start
            step = 1 if id.step == None else id.step
            return [self[i] for i in range(start, id.stop, step)]
        elif isinstance(id, list):
            return [self[i] for i in id]
        else:
            raise TypeError("Invalid argument type.")

    def _read_data(cls, fpath):
        sents = list()
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                sents.append(line.rstrip("\n"))

        return sents

    # def _create_parallel_vocab(self):


    # def tokenize_pair(self, id):
    #     src = self.parallel_vocab.tokenize_corpus([self.src_sents[id]])
    #     tgt = self.parallel_vocab.tokenize_corpus([self.tgt_sents[id]], is_src=False)
    #     return {"src": src, "tgt": tgt}