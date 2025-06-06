"""
Source: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

from torch import nn


class FFNN(nn.Module):

    def __init__(
            self,
            d,
            bias=False,
            dropout=0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        bias: whether to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.c_fc = nn.Linear(d, 4 * d, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d, d, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)  # [B, T, 4*d]
        x = self.gelu(x)
        x = self.c_proj(x)  # [B, T, d]
        x = self.dropout(x)
        return x
