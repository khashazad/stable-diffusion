import attention
import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)

        self.positional_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:

        # batch_size x sequence_length -> batch_size x sequence_length x Dim'
        x = self.token_embedding(tokens)

        x += self.positional_embedding

        return x

class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embed: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)

        self.linear_1 = nn.Linear(n_embed, n_embed * 4)
        self.linear_2 = nn.Linear(n_embed * 4, n_embed)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:


        ### Attention

        residual = x

        x = self.layernorm_1(x)

        x = self.attention(x, causal_mask=None)

        x += residual

        ### Feed-Forward

        residual = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x) # Quick GELU

        x = self.linear_2(x)

        x += residual

        return x

class CLIP(nn.Module):

    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for _ in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:

        tokens = tokens.type(torch.long)

        # batch_size x sequence_length -> batch_size x sequence_length x 768
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # batch_size x sequence_length x 768
        output = self.layernorm(state)

        return output
