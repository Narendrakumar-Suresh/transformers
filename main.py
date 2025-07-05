import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import Transformer
from data.data import train_loader, valid_loader, test_loader, src_vocab, tgt_vocab, tokenize_de
import time

# ==== Constants ====
VOCAB_SIZE = len(tgt_vocab)  # Use actual vocabulary size from data
SRC_VOCAB_SIZE = len(src_vocab)
PAD_IDX = 0
BATCH_SIZE = 32
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 1024
NUM_LAYERS = 6
MAX_LEN = 100
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOS_IDX = 1
EOS_IDX = 2

print(f"Source vocabulary size: {SRC_VOCAB_SIZE}")
print(f"Target vocabulary size: {VOCAB_SIZE}")
print(f"Using device: {DEVICE}")

# === Training Functions ===
def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (src_batch, tgt_batch) in enumerate(data_loader):
        src_batch = src_batch.to(DEVICE)
        tgt_batch = tgt_batch.to(DEVICE)
        
        # Create target input (shifted by 1 for teacher forcing)
        tgt_input = tgt_batch[:, :-1]  # Remove EOS token
        tgt_output = tgt_batch[:, 1:]  # Remove SOS token
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(src_batch, tgt_input)
        
        # Reshape for loss calculation
        logits = logits.reshape(-1, VOCAB_SIZE)
        tgt_output = tgt_output.reshape(-1)

        
        loss = criterion(logits, tgt_output)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for src_batch, tgt_batch in data_loader:
            src_batch = src_batch.to(DEVICE)
            tgt_batch = tgt_batch.to(DEVICE)
            
            # Create target input (shifted by 1)
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            
            # Forward pass
            logits = model(src_batch, tgt_input)
            
            # Reshape for loss calculation
            logits = logits.reshape(-1, VOCAB_SIZE)
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(logits, tgt_output)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def translate_sentence(model, sentence, max_len=50):
    """Translate a German sentence to English using greedy decoding"""
    model.eval()
    
    # Tokenize the input sentence
    tokens = tokenize_de(sentence.lower())
    tokens = ["<sos>"] + tokens + ["<eos>"]
    
    # Convert to indices using the vocabulary's encode method
    src_indices = src_vocab.encode(tokens)
    src_tensor = torch.tensor(src_indices).unsqueeze(0).to(DEVICE)
    
    # Initialize target sequence with BOS
    tgt_indices = [BOS_IDX]
    
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(DEVICE)
            
            # Get predictions
            output = model(src_tensor, tgt_tensor)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Add to sequence
            tgt_indices.append(next_token.item())
            
            # Stop if EOS token is generated
            if next_token.item() == EOS_IDX:
                break
    
    # Convert back to tokens using the vocabulary's decode method
    tgt_tokens = tgt_vocab.decode(tgt_indices[1:-1])  # Remove BOS and EOS
    return " ".join(tgt_tokens)

def train():
    # Initialize model with actual vocabulary sizes
    model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_layers=NUM_LAYERS
).to(DEVICE)



    
    print(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    best_valid_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        
        # Validation
        valid_loss = evaluate(model, valid_loader, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "transformer_translation_best.pth")
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.4f}')
        print(f'\tValid Loss: {valid_loss:.4f}')
        print(f'\tBest Valid Loss: {best_valid_loss:.4f}')
        
        # Sample translation after each epoch
        if epoch % 5 == 0:
            sample_sentence = "Ein Mann läuft auf der Straße."
            translation = translate_sentence(model, sample_sentence)
            print(f'\tSample translation: "{sample_sentence}" -> "{translation}"')
    
    # Save final model
    torch.save(model.state_dict(), "transformer_translation_final.pth")
    return model

if __name__ == '__main__':
    # Train the model
    model = train()
    
    # Load best model for evaluation
    model.load_state_dict(torch.load("transformer_translation_best.pth", map_location=DEVICE))
    
    # Test the model
    test_loss = evaluate(model, test_loader, nn.CrossEntropyLoss(ignore_index=PAD_IDX))
    print(f'Test Loss: {test_loss:.4f}')
    
    # Example translations
    print("\n" + "="*60)
    print("EXAMPLE TRANSLATIONS")
    print("="*60)
    
    example_sentences = [
        "Ein Mann läuft auf der Straße.",
        "Das Wetter ist heute schön.",
        "Ich gehe zur Schule.",
        "Das Buch liegt auf dem Tisch.",
        "Der Hund spielt im Garten.",
        "Die Frau liest ein Buch.",
        "Das Kind spielt mit dem Ball.",
        "Der Mann fährt Auto."
    ]
    
    for sentence in example_sentences:
        translation = translate_sentence(model, sentence)
        print(f'German: {sentence}')
        print(f'English: {translation}')
        print()
    
    print("Training and evaluation completed!")