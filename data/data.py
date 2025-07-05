from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spacy_de = get_tokenizer("spacy", language="de_core_news_sm")
spacy_en = get_tokenizer("spacy", language="en_core_web_sm")

def yield_tokens(data_iter, tokenizer):
    for src, tgt in data_iter:
        yield tokenizer(src)

# Load training data for vocab
train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

# Build vocabularies
src_vocab = build_vocab_from_iterator(yield_tokens(train_iter, spacy_de), specials=['<pad>', '<sos>', '<eos>', '<unk>'])
src_vocab.set_default_index(src_vocab['<unk>'])

train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))  # reload it
tgt_vocab = build_vocab_from_iterator(yield_tokens(((tgt, src) for src, tgt in train_iter), spacy_en), specials=['<pad>', '<sos>', '<eos>', '<unk>'])
tgt_vocab.set_default_index(tgt_vocab['<unk>'])

def tokenize_and_encode(src, tgt):
    src_tensor = torch.tensor([src_vocab['<sos>']] + [src_vocab[token] for token in spacy_de(src)] + [src_vocab['<eos>']], dtype=torch.long)
    tgt_tensor = torch.tensor([tgt_vocab['<sos>']] + [tgt_vocab[token] for token in spacy_en(tgt)] + [tgt_vocab['<eos>']], dtype=torch.long)
    return src_tensor, tgt_tensor

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_tensor, tgt_tensor = tokenize_and_encode(src_sample, tgt_sample)
        src_batch.append(src_tensor)
        tgt_batch.append(tgt_tensor)
    src_batch = pad_sequence(src_batch, padding_value=src_vocab['<pad>'], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_vocab['<pad>'], batch_first=True)
    return src_batch.to(DEVICE), tgt_batch.to(DEVICE)

train_data = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
valid_data = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
test_data  = Multi30k(split='test', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

train_loader = DataLoader(list(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(list(valid_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(list(test_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
