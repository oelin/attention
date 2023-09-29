"""Scaled Dot-product Attention.

This implementation is based on PyTorch's `scaled_dot_product_attention` 
function, which takes in precomputed query, key and value tensors as well as 
an optional mask tensor. If `is_causal` is true, then a causal maks is used.

See: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
See: https://paperswithcode.com/method/scaled
"""

from typing import Optional
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    mask: Optional[torch.Tensor] = None, 
    dropout_probability: float = 0.0, 
    is_causal: bool = False,
) -> torch.Tensor:
    """Scaled Dot-product Attention."""

    query_batch_size, query_sequence_length, query_dimension = query.size()
    key_batch_size, key_sequence_length, key_dimension = query.size()
    value_batch_size, value_sequence_length, value_dimension = value.size()

    # Checks (can be omitted).

    assert query_batch_size == key_batch_size
    assert query_batch_size == value_batch_size

    if mask is not None:        
        mask_query_sequence_length, mask_key_sequence_length = mask.size()
        
        assert not is_causal
        assert query_sequence_length == mask_query_sequence_length
        assert key_sequence_length == mask_key_sequence_length

    assert key_sequence_length == value_sequence_length
    assert query_dimension == key_dimension

    # Computation.

    if is_causal:
        mask = torch.ones(query_sequence_length, key_sequence_length, dtype=torch.float).tril(diagonal=0)
        mask = mask.masked_fill(mask == 0, -torch.inf)

    score = query @ key.transpose(-2, -1)
    score = score / math.sqrt(query_dimension)
    score = score + mask
    score = torch.softmax(score, dim=-1)
    output = score @ V
    
    return output
    

class ScaledDotProductAttention(nn.Modlue):
    """Scaled Dot-product Attention."""

    def __init__(self) -> None:
        super().__init__()
    
    def forward(
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        dropout_probability: float = 0.0, 
        is_causal: bool = False,
    ) -> torch.Tensor

        return scaled_dot_product_attention(
            query, 
            key, 
            mask, 
            dropout_probability, 
            is_causal,
        )
