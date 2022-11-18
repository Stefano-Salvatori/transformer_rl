from torch import nn


class CriticModel(nn.Module):
    def __init__(self, config, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.activation = nn.Tanh()
        self.linear1 = nn.Linear(config.hidden_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states):
        output = self.linear1(hidden_states)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.activation(output)
        output = self.final(output)
        return output
