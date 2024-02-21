
class CrossAttention(nn.Module):
    """Cross attention.

    Example
    -------
    >>> module = CrossAttention(embedding_dimension=256, heads=16)
    >>> x = torch.randn((1, 10, 256))
    >>> y = torch.randn((1, 20, 256))
    >>> x = module(x, y)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
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
            out_features=embedding_dimension,
            bias=False,
        )

        self.linear_2 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )

        self.linear_2 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )

        self.linear_4 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The target input tensor.
        y : torch.Tensor
            The source input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        q = rearrange(self.linear_1(x), 'b t (h e) -> b h t e', h=self.heads)
        k = rearrange(self.linear_2(y), 'b s (h e) -> b h s e', h=self.heads)
        v = rearrange(self.linear_3(y), 'b s (h e) -> b h s e', h=self.heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = self.linear_4(rearrange(x, 'b h t e -> b t (h e)'))

        return x
