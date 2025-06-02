import dataclasses
from .constants import VOCAB_SIZE


@dataclasses.dataclass
class Config:
    embedding_dim = 512
    hidden_dim = 512
    num_layers = 4
    vocab_size = VOCAB_SIZE