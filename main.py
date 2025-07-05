import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import Transformer

# ==== Constants ====
VOCAB_SIZE = 100
SEQ_LEN = 10
PAD_IDX = 0
BATCH_SIZE = 32
EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_LAYERS = 2
MAX_LEN = 20
EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOS_IDX = 1

# === Helper Functions ===
def generate_batch(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE):
    x = torch.randint(2, vocab_size, (batch_size, seq_len))
    y = torch.flip(x, dims=[1])
    return x.to(DEVICE), y.to(DEVICE)

def evaluate(model, input_seq):
    model.eval()
    with torch.no_grad():
        input_seq = input_seq.unsqueeze(0).to(DEVICE)  # (1, T)
        tgt_seq = torch.full((1, 1), BOS_IDX, dtype=torch.long, device=DEVICE)  # <BOS>

        for _ in range(input_seq.size(1)):
            output = model(input_seq, tgt_seq)  # masking is handled inside Transformer
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt_seq = torch.cat([tgt_seq, next_token], dim=1)

        return tgt_seq[0, 1:]  # strip <BOS>

def train():
    model = Transformer(VOCAB_SIZE, max_len=MAX_LEN, embed_dim=EMBED_DIM,
                        num_heads=NUM_HEADS, ff_dim=FF_DIM, num_layers=NUM_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(EPOCHS):
        model.train()
        x, y = generate_batch()
        tgt_input = torch.cat([torch.full((BATCH_SIZE, 1), BOS_IDX, device=DEVICE), y[:, :-1]], dim=1)

        logits = model(x, tgt_input)  # masks handled internally
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "transformer_reverse.pth")
    return model

if __name__ == '__main__':
    model = train()

    # Test example
    test_input = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    reversed_output = evaluate(model, test_input)
    print("Input:", test_input.tolist())
    print("Reversed Prediction:", reversed_output.tolist())
