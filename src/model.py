import torch.nn as nn

class ShakespeareModel(nn.Module):
    def __init__(self, char_size, dim_embed=10, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(char_size, embedding_dim=dim_embed)
        self.gru = nn.GRU(
            dim_embed,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True, #para ratificar que vai esperar [batch/window_len/embed_size]
            dropout=0.2
        )
        self.output = nn.Linear(hidden_dim, char_size)

    def forward(self, X):
        embeddings = self.embed(X) #crio os embeddings
        outputs, h_n = self.gru(embeddings)
        return self.output(outputs).permute(0, 2, 1) 

        #shape sem permute -> [batch size, window length, vocabulary size]
        #shape com permute -> [batch size, vocabulary size, window length]