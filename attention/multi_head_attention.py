def multi_head_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    WQ: torch.Tensor,
    WK: torch.Tensor,
    WV: torch.Tensor,
    W0: torch.Tensor,
    mask: torch.Tensor,
    embedding_dimension: int,
    number_of_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-head (scaled dot-product) attention.

    Parameters
    ----------

    Q: torch.Tensor - Query tensor of shape (B, T, QD).
    K: torch.Tensor - Key tensor of shape (B, S, QD).
    V: torch.Tensor - Value tensor of shape (B, S, VD).
    WQ: torch.Tensor - Query projection tensor of shape (ED, QD).
    WK: torch.Tensor - Key projection tensor of shape (ED, ED).
    WV: torch.Tensor - Value projection tensor of shape (ED, VD).
    W0: torch.Tensor - Output projection tensor of shape (ED, ED).
    mask: torch.Tensor - Mask tensor of shape (T, S).
    embedding_dimension: int - Embedding dimension.
    number_of_heads: int - Number of heads.

    Returns
    -------

    score: torch.Tensor - attention score of shape (B, N, T, S).
    value: torch.Tensor - attention value of shape (B, T, VD).

    Example
    -------

    >>> Q = torch.randn(5, 10, 15)
    >>> K = torch.randn(5, 20, 15)
    >>> V = torch.randn(5, 20, 25)
    >>> WQ = torch.randn(15, 15)
    >>> WK = torch.randn(15, 15)
    >>> WV = torch.randn(25, 15)
    >>> W0 = torch.randn(15, 15)
    >>> mask = torch.ones((10, 20))
    >>> score, value = multi_head_attention(Q, K, V, WQ, WK, WV, W0, mask, 3)

    Aliases
    -------

    B - batch size.
    T - target sequence length.
    S - source sequence length.
    QD - query dimension.
    VD - value dimension.
    ED - embedding dimension.
    """

    head_dimension = embedding_dimension // number_of_heads
    target_sequence_length = Q.size(1)
    source_sequence_length = K.size(1)

    Q = Q @ WQ.T  
    K = K @ WK.T
    V = V @ WV.T

    Q = Q.view(-1, number_of_heads, target_sequence_length, head_dimension)
    K = K.view(-1, number_of_heads, source_sequence_length, head_dimension)
    V = V.view(-1, number_of_heads, source_sequence_length, head_dimension)

    value, score = scaled_dot_product_attention(Q, K, V, mask)

    # Concatenate heads.

    value = value.permute(2, 0, 1, 3)
    value = value.contiguous()
    value = value.transpose(0, 1)
    value = value.view(-1, target_sequence_length, embedding_dimension)
    value = value @ W0.T  

    return value, score
