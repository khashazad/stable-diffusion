import math
import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, num_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()

        self.in_projection = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)
        self.out_projection = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.num_heads = num_heads
        self.d_head = d_embed // num_heads

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        # x: batch_size, sequence_length, dimension

        input_shape = x.shape

        batch_size, seq_len, d_embed = input_shape

        intermim_shape = (batch_size, seq_len, self.num_heads, self.d_head)

        # batch_size, sequence_length, dimension -> batch_size, sequence_length, 3 * dimension -> 3 tensors of shape (batch_size, sequence_length, num_heads, d_head)
        q, k, v = self.in_projection(x).chunk(3, dim=-1)

        # batch_size, sequence_length, d_embed > batch_size, sequence_length, H, d_embed // H -> batch_size, H, sequence_length, d_embed // H
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        # batch_size, H, seq_len, seq_len
        attention_weights = k @ q.transpose(-2, -1)

        if causal_mask:
            # mask where the upper triangle (above the diagonal) is made up of 1
            mask = torch.ones_like(attention_weights, dtype=torch.bool).triu(1)

            attention_weights.masked_fill_(mask, -torch.inf)

        attention_weights /= math.sqrt(self.d_head)

        # batch_size, H, seq_len, seq_len
        attention_scores = attention_weights.softmax(dim=-1)

        # batch_size, H, seq_len, d_head
        output = attention_scores @ v

        output = output.transpose(1, 2).reshape(input_shape)

        # batch_size, sequence_length, dimension
        return self.out_projection(output)
