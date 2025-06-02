from torch import nn
from torch.nn import functional as F

from .config import Config

class LSTMDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim, config.num_layers, batch_first=True)
        self.fc = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, video_features, captions):
        embeddings = self.embed(captions)

        # Expand hidden states to match num_layers
        h0 = video_features.unsqueeze(0).repeat(self.config.num_layers, 1, 1)  # (num_layers, batch, hidden_dim)
        c0 = video_features.unsqueeze(0).repeat(self.config.num_layers, 1, 1)

        out, _ = self.lstm(embeddings, (h0, c0))
        logits = self.fc(out)
        return logits