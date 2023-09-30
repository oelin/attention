from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.scaled_dot_product_attention import scaled_dot_product_attention, ScaledDotProductAttention


class SlowMultiHeadAttention(nn.Module):
    """Slow multi-head attention. 

    Uses `number_of_heads` ScaledDotProductAttention layers to compute the output
    of each head, then concatenates them.
    """
    
    def __init__(
        self, 
        number_of_heads: int, 
        head_dimension: int, 
        output_dimension: int,
    ) -> None:
        super().__init__()

        self.linear_output = nn.Linear(
            head_dimension * number_of_heads, 
            output_dimension, 
            bias=False,
        )
        
        self.heads = nn.ModuleList([
            ScaledDotProductAttention(
                query_dimension=head_dimension, 
                value_dimension=head_dimension,
            )

            for _ in range(number_of_heads)
        ])

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = torch.cat([head(query, key, value, mask) for head in self.heads], dim=-1)
        x = self.linear_output(x)

        return x


class MultiHeadAttention(nn.Module):
    """Efficient multi-head attention."""

    ...
