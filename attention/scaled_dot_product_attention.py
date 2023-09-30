def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention.

    Parameters
    ----------

    Q: torch.Tensor - Query tensor of shape (B, T, QD).
    K: torch.Tensor - Key tensor of shape (B, S, QD).
    V: torch.Tensor - Value tensor of shape (B, S, VD).
    mask: torch.Tensor - Mask tensor of shape (T, S).

    Returns
    -------

    value: torch.Tensor - attention value of shape (B, T, VD).
    score: torch.Tensor - attention score of shape (T, S).

    Example
    -------

    >>> Q = torch.randn(5, 10, 15)
    >>> K = torch.randn(5, 20, 15)
    >>> V = torch.randn(5, 20, 25)
    >>> mask = torch.ones((10, 20))
    >>> score, value = scaled_dot_product_attention(Q, K, V, mask) 

    Aliases
    -------

    B - batch size.
    T - target sequence length.
    S - source sequence length.
    QD - query dimension.
    VD - value dimension.
    """

    mask = mask.masked_fill(mask == 0, -torch.inf)
    score = Q @ K.transpose(-2, -1)
    score = score / math.sqrt(Q.size(-1))
    score = score + mask
    score = torch.softmax(score, dim=-1)
    value = score @ V

    return value, score
