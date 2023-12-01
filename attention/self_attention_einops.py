import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class Attention(nn.Module):
    """Attention."""

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
    ) -> None:
        """Initialize the module."""

        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads 

        self.linear_1 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension * 3,
            bias=False,
        )

        self.linear_2 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        q, k, v = rearrange(self.linear_1(x), 'b s (k h e) -> k b h s e', k=3, h=self.number_of_heads)

        score = torch.einsum('bhxe,bhye->bhxy', q, k)
        score = F.softmax((score / math.sqrt(k.size(-1))) + mask, dim=-1)

        x = rearrange(score @ v, 'b h s e -> b s (h e)')

        return x
