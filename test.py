import torch 

from models.model.transformer import Transformer
import utils
import utils.data_utils as data_utils
import utils.model_utils as model_utils
from utils.bleu import get_bleu, idx_to_word


# Load config file
config = data_utils.get_config("config.yml")

for key, value in config.items():
    globals()[key] = value


envi_vocab = torch.load(path["parallel_vocab"])
dataloaders = torch.load(path["dataloaders"])
test_loader = dataloaders["test_loader"]


# Load model
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
                    drop_prob=0.00,
                    device=device).to(device)

print(f'The model has {model_utils.count_parameters(model):,} trainable parameters')


def test_model(parallel_vocab):
    iterator = test_loader

    model.load_state_dict(torch.load("./saved/model-saved.pt"))

    with torch.no_grad():
        batch_bleu = []
        for i, batch in enumerate(iterator):
            src = batch["src"].to(device)
            trg = batch["tgt"].to(device)
            output = model(src, trg[:, :-1])

            total_bleu = []
            for j in range(batch["tgt"].shape[0]):
                src_words = idx_to_word(src[j], parallel_vocab.src)
                trg_words = idx_to_word(trg[j], parallel_vocab.tgt)
          
                output_words = output[j].max(dim=1)[1]
                output_words = idx_to_word(output_words, parallel_vocab.tgt)

                print('source :', src_words)
                print('target :', trg_words)
                print('predicted :', output_words)
                print()
                bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                total_bleu.append(bleu)


            total_bleu = sum(total_bleu) / len(total_bleu)
            print('BLEU SCORE = {}'.format(total_bleu))
            batch_bleu.append(total_bleu)

        batch_bleu = sum(batch_bleu) / len(batch_bleu)
        print('TOTAL BLEU SCORE = {}'.format(batch_bleu))


if __name__ == '__main__':
    test_model(envi_vocab)