import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read, then decode for py2 compat.
with open('samples.txt', 'r') as f:
    text = f.read()

# The unique characters in the file
vocab = sorted(set(text))

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters
seq_length = 100

# Create training examples / targets
def create_sequences(text_as_int, seq_length):
    examples_per_epoch = len(text)//(seq_length+1)
    char_dataset = []
    
    for i in range(0, examples_per_epoch):
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

# Batch size
BATCH_SIZE = 64

dataset = CharDataset(sequences)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, rnn_units, batch_first=True)
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = self.fc(x)

        return x

model = Model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

EPOCHS = 10

for epoch in range(EPOCHS):
    for batch, (input_example, target_example) in enumerate(dataloader):
        input_example = input_example.to(device)
        target_example = target_example.to(device)

        # forward pass
        output = model(input_example)
        loss = criterion(output.transpose(1, 2), target_example)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Save the model after training
torch.save(model.state_dict(), 'model.pth')

# Load the model for generating text
model = Model(vocab_size, embedding_dim, rnn_units, 1).to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

def generate_text(model, start_string, num_generate = 10000):
    # Converting our start string to numbers (vectorizing)
    input_eval = torch.tensor([char2idx[s] for s in start_string]).unsqueeze(0).to(device)

    # Empty string to store our results
    text_generated = []

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = predictions.squeeze(0).cpu().detach().numpy()

        # using a categorical distribution to predict the character returned by the model
        predicted_id = np.random.choice(range(vocab_size), p=np.exp(predictions[-1]))

        # We pass the predicted character as the next input to the model
        input_eval = torch.tensor([[predicted_id]], device=device)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"skalthoff says "))
