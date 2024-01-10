
class RoPE(nn.Module):
    """Rotary positional embedding (RoPE).

    Rotary positional embedding (Su et al., 2023) rotates keys and queries by
    their absolute position such that their dot product depends only on their
    content and *relative position*. Generalized to arbitrary dimensions, RoPE
    divides a D-dimensional space into D//2 subspaces.

    Example
    -------
    >>> module = RoPE(embedding_dimension=256, base=10_000)
    >>> q = torch.randn((1, 10, 256))
    >>> k = torch.randn((1, 10, 256))
    >>> alignment = torch.einsum('bte,bse->bts', module(q), module(k))
    """

    def __init__(self, *, embedding_dimension: int, base: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        base : int
            The base to use for absolute positional encodings.
        """

        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.base = base

        # Precompute theta.

        exponent = torch.arange(
            start=0,
            end=embedding_dimension,
            step=2,
            dtype=torch.float,
        ) / embedding_dimension

        theta = 1. / torch.pow(base, exponent)

        self.theta = theta

    def absolute_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Perform absolute positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        encoding : torch.Tensor
            The absolute positional encoding.
        """

        if self.theta.device != x.device:
            self.theta = self.theta.to(x.device)

        encoding = torch.einsum(
            't,e->te',
            torch.arange(x.size(-2), dtype=torch.float, device=x.device),
            self.theta,
        )

        encoding = repeat(encoding, '... e -> ... (e n)', n=2)

        return encoding

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate each subspace by -90 degrees."""

        x = rearrange(x, '... (e n) -> ... e n', n=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        x = rearrange(x, '... e n -> ... (e n)')

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Foward pass."""

        encoding = self.absolute_positional_encoding(x)
        x = x * encoding.cos() + (self.rotate_half(x) * encoding.sin())

        return x


class Attention(nn.Module):
    """Attention.

    Implements multi-head self attention (Vaswani et al., 2017) with rotary
    positional embedding (RoPE) (Lu et al., 2021).

    Example
    -------
    >>> module = Attention(
    ...    embedding_dimension=256,
    ...    number_of_heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """
    
    def __init__(
        self, 
        *,
        embedding_dimension: int, 
        number_of_heads: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        number_of_heads : int
            The number of heads.
        """
        
        super().__init__()

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

        self.rope = RoPE(
            embedding_dimension=embedding_dimension // number_of_heads,
            base=10_000,
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        h = self.number_of_heads
        q, k, v = rearrange(self.linear_1(x), 'b s (n h e) -> n b h s e', n=3, h=h)
        q, k = self.rope(q), self.rope(k)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = self.linear_2(rearrange(x, 'b h s e -> b s (h e)'))
      
        return x
