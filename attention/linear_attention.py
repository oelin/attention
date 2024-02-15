def linear_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
) -> torch.Tensor:

    score = F.softmax(torch.einsum('bhnk,bhnc->bhkc', k/4, v), dim=-1)
    x = F.softmax(torch.einsum('bhnk,bhkc->bhnk', q/4, score), dim=-1)

    return x


class LinearAttention(nn.Module):
    """Attention.

    Example
    -------
    >>> module = Attention(
    ...    embedding_dimension=256,
    ...    heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(
        self,
        *,
        embedding_dimension: int,
        heads: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of heads.
        """

        super().__init__()

        self.heads = heads

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        q, k, v = rearrange(self.linear_1(x), 'b s (n h e) -> n b h s e', n=3, h=self.heads)
        x = linear_attention(q, k, v) #F.scaled_dot_product_attention(q, k, v)
        x = self.linear_2(rearrange(x, 'b h s e -> b s (h e)'))

        return x
