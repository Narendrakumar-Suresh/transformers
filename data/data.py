import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import urllib.request
import gzip
import os

# Simple tokenizers
def tokenize_de(text):
    return text.lower().split()

def tokenize_en(text):
    return text.lower().split()

class Vocabulary:
    def __init__(self, max_size=10000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        
    def build_vocab(self, texts, tokenizer):
        counter = Counter()
        for text in texts:
            tokens = tokenizer(text)
            counter.update(tokens)
        
        # Sort by frequency and take top max_size - 4 (for special tokens)
        most_common = counter.most_common(self.max_size - 4)
        
        # Add tokens that meet minimum frequency requirement
        for token, freq in most_common:
            if freq >= self.min_freq:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token
    
    def __len__(self):
        return len(self.stoi)
    
    def encode(self, tokens):
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]
    
    def decode(self, indices):
        return [self.itos[idx] for idx in indices]

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        # Tokenize
        src_tokens = self.src_tokenizer(src_text)
        tgt_tokens = self.tgt_tokenizer(tgt_text)
        
        # Add special tokens
        src_tokens = ["<sos>"] + src_tokens + ["<eos>"]
        tgt_tokens = ["<sos>"] + tgt_tokens + ["<eos>"]
        
        # Convert to indices
        src_indices = self.src_vocab.encode(src_tokens)
        tgt_indices = self.tgt_vocab.encode(tgt_tokens)
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

def download_file(url, filename):
    """Download file if it doesn't exist"""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")

def load_multi30k_data():
    """Load Multi30k dataset manually"""
    # URLs for Multi30k dataset
    base_url = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
    
    files = {
        "train.de": f"{base_url}train.de.gz",
        "train.en": f"{base_url}train.en.gz",
        "val.de": f"{base_url}val.de.gz",
        "val.en": f"{base_url}val.en.gz",
        "test_2016_flickr.de": f"{base_url}test_2016_flickr.de.gz",
        "test_2016_flickr.en": f"{base_url}test_2016_flickr.en.gz",
    }
    
    # Create data directory
    os.makedirs("data_files", exist_ok=True)
    
    # Download files
    for filename, url in files.items():
        gz_path = os.path.join("data_files", filename + ".gz")
        txt_path = os.path.join("data_files", filename)
        
        # Download if not exists
        if not os.path.exists(gz_path) and not os.path.exists(txt_path):
            try:
                download_file(url, gz_path)
                
                # Extract gzipped file
                with gzip.open(gz_path, 'rt', encoding='utf-8') as f_in:
                    with open(txt_path, 'w', encoding='utf-8') as f_out:
                        f_out.write(f_in.read())
                
                # Remove gzipped file
                os.remove(gz_path)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                # Create dummy data for testing
                create_dummy_data(txt_path)
    
    # Read the files
    def read_file(filename):
        filepath = os.path.join("data_files", filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines()]
        else:
            return create_dummy_data(filepath)
    
    train_de = read_file("train.de")
    train_en = read_file("train.en")
    val_de = read_file("val.de")
    val_en = read_file("val.en")
    test_de = read_file("test_2016_flickr.de")
    test_en = read_file("test_2016_flickr.en")
    
    return (train_de, train_en), (val_de, val_en), (test_de, test_en)

def create_dummy_data(filepath):
    """Create dummy data for testing when download fails"""
    print(f"Creating dummy data for {filepath}")
    
    # German sentences
    dummy_de = [
        "Ein Mann läuft auf der Straße.",
        "Das Wetter ist heute schön.",
        "Ich gehe zur Schule.",
        "Das Buch liegt auf dem Tisch.",
        "Der Hund spielt im Garten.",
        "Die Frau liest ein Buch.",
        "Das Kind spielt mit dem Ball.",
        "Der Mann fährt Auto.",
        "Die Katze schläft auf dem Sofa.",
        "Es regnet heute sehr stark."
    ] * 100  # Repeat to have more data
    
    # English sentences
    dummy_en = [
        "A man is running on the street.",
        "The weather is nice today.",
        "I am going to school.",
        "The book is on the table.",
        "The dog is playing in the garden.",
        "The woman is reading a book.",
        "The child is playing with the ball.",
        "The man is driving a car.",
        "The cat is sleeping on the sofa.",
        "It is raining very hard today."
    ] * 100  # Repeat to have more data
    
    # Determine which type of data to create
    if "de" in filepath:
        data = dummy_de
    else:
        data = dummy_en
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')
    
    return data

# Load the data
print("Loading Multi30k dataset...")
try:
    (train_src, train_tgt), (valid_src, valid_tgt), (test_src, test_tgt) = load_multi30k_data()
except Exception as e:
    print(f"Error loading data: {e}")
    print("Using dummy data for testing...")
    # Create dummy data
    train_src = create_dummy_data("dummy_train.de")
    train_tgt = create_dummy_data("dummy_train.en")
    valid_src = train_src[:100]
    valid_tgt = train_tgt[:100]
    test_src = train_src[:50]
    test_tgt = train_tgt[:50]

print(f"Training data: {len(train_src)} German sentences, {len(train_tgt)} English sentences")
print(f"Validation data: {len(valid_src)} German sentences, {len(valid_tgt)} English sentences")
print(f"Test data: {len(test_src)} German sentences, {len(test_tgt)} English sentences")

# Build vocabularies
print("Building vocabularies...")
src_vocab = Vocabulary(max_size=10000, min_freq=2)
tgt_vocab = Vocabulary(max_size=10000, min_freq=2)

src_vocab.build_vocab(train_src, tokenize_de)
tgt_vocab.build_vocab(train_tgt, tokenize_en)

print(f"Unique tokens in source (de) vocabulary: {len(src_vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(tgt_vocab)}")

# Create datasets
train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab, tokenize_de, tokenize_en)
valid_dataset = TranslationDataset(valid_src, valid_tgt, src_vocab, tgt_vocab, tokenize_de, tokenize_en)
test_dataset = TranslationDataset(test_src, test_tgt, src_vocab, tgt_vocab, tokenize_de, tokenize_en)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

print(f"Number of training examples: {len(train_dataset)}")
print(f"Number of validation examples: {len(valid_dataset)}")
print(f"Number of testing examples: {len(test_dataset)}")

# Test the data loader
print("Testing data loader...")
for src_batch, tgt_batch in train_loader:
    print(f"Source batch shape: {src_batch.shape}")
    print(f"Target batch shape: {tgt_batch.shape}")
    print(f"Sample source sentence: {src_vocab.decode(src_batch[0].tolist())}")
    print(f"Sample target sentence: {tgt_vocab.decode(tgt_batch[0].tolist())}")
    break

print("Data loading completed successfully!")