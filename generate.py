import torch
import torch.nn as nn
import numpy as np
import torch_directml

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


with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.encode("ascii", errors="ignore").decode()

vocab = sorted(set(text))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

# Create DirectML device
device = torch_directml.device()

# Load model
model = Model(vocab_size, embedding_dim, rnn_units)
model.load_state_dict(torch.load('model.pth'))
model.to(device)
model.eval()

import torch.nn.functional as F

def generate_text(model, start_string, num_generate=1000, temperature=0.2):
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
