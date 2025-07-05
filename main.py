import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import Transformer
from data.data import train_loader, valid_loader, test_loader, src_vocab, tgt_vocab
import time

# ==== Constants ====
VOCAB_SIZE = len(tgt_vocab)
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

        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]
        
        optimizer.zero_grad()
        logits = model(src_batch, tgt_input)
        logits = logits.reshape(-1, VOCAB_SIZE)
        tgt_output = tgt_output.reshape(-1)
        
        loss = criterion(logits, tgt_output)
        loss.backward()
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
            
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            logits = model(src_batch, tgt_input)
            logits = logits.reshape(-1, VOCAB_SIZE)
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(logits, tgt_output)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train():
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
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        valid_loss = evaluate(model, valid_loader, criterion)
        end_time = time.time()
        mins, secs = divmod(end_time - start_time, 60)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "transformer_translation_best.pth")

        print(f'Epoch: {epoch+1:02} | Time: {int(mins)}m {secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.4f}')
        print(f'\tValid Loss: {valid_loss:.4f}')
        print(f'\tBest Valid Loss: {best_valid_loss:.4f}')

    torch.save(model.state_dict(), "transformer_translation_final.pth")
    return model

if __name__ == '__main__':
    model = train()

    # Load best model for final evaluation
    model.load_state_dict(torch.load("transformer_translation_best.pth", map_location=DEVICE))
    model.eval()

    test_loss = evaluate(model, test_loader, nn.CrossEntropyLoss(ignore_index=PAD_IDX))
    print(f'Test Loss: {test_loss:.4f}')

    # === Export to ONNX ===
    print("Exporting to ONNX...")
    src_dummy = torch.randint(0, SRC_VOCAB_SIZE, (1, 10)).to(DEVICE)  # (B, T_src)
    tgt_dummy = torch.randint(0, VOCAB_SIZE, (1, 9)).to(DEVICE)       # (B, T_tgt)

    torch.onnx.export(
        model,
        (src_dummy, tgt_dummy),
        "transformer_translation.onnx",
        input_names=["src_ids", "tgt_ids"],
        output_names=["logits"],
        dynamic_axes={"src_ids": {1: "src_seq_len"}, "tgt_ids": {1: "tgt_seq_len"}, "logits": {1: "tgt_seq_len"}},
        opset_version=17,
        do_constant_folding=True
    )
    print("ONNX export complete: transformer_translation.onnx")
