import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch_directml # Import DirectML

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.encode("ascii", errors="ignore").decode()

vocab = sorted(set(text))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

seq_length = 150

def create_sequences(text_as_int, seq_length):
    num_sequences = len(text)//(seq_length+1)
    char_dataset = []
    
    for i in range(0, num_sequences):
        char_dataset.append(text_as_int[i: i + seq_length + 1])
    
    return char_dataset

sequences = create_sequences(text_as_int, seq_length)

class CharDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return sequence[:-1], sequence[1:]

BATCH_SIZE = 16

dataset = CharDataset(sequences)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=6)

vocab_size = len(vocab)
embedding_dim = 128
rnn_units = 4096

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, rnn_units, batch_first=True)

        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.gru(x, hidden)
        x = self.fc(x)

        return x, hidden

# Create DirectML device
device = torch_directml.device()

model = Model(vocab_size, embedding_dim, rnn_units).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

EPOCHS = 100
best_loss = float('inf')

for epoch in range(EPOCHS):
    hidden = None

    for batch, (input_example, target_example) in enumerate(dataloader):
        input_example = input_example.to(device)
        target_example = target_example.to(device)

        output, hidden = model(input_example, hidden)
        hidden = hidden.detach()

        loss = criterion(output.transpose(1, 2), target_example)

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        # Model checkpointing
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'model.pth')
    
    # Learning rate scheduler
    scheduler.step(loss)

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

import torch.nn.functional as F

def generate_text(model, start_string, num_generate = 10000, temperature=0.8):
    input_eval = torch.tensor([char2idx[s] for s in start_string]).unsqueeze(0).to(device)
    text_generated = []
    hidden = None

    for i in range(num_generate):
        predictions, hidden = model(input_eval, hidden)
        predictions = predictions.squeeze(0).cpu().detach().numpy()
        probabilities = F.softmax(torch.from_numpy(predictions[-1]/temperature), dim=0).numpy()

        predicted_id = np.random.choice(range(vocab_size), p=probabilities)
        input_eval = torch.tensor([[predicted_id]], device=device)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


print(generate_text(model, start_string=u"skalthoff says "))