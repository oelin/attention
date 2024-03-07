def frequency_encode_1d(
    x: torch.Tensor,
    embedding_dimension: int,
    base: float = 1e4,
) -> torch.Tensor:
    """Freqency encode a 1D index tensor."""

    half = embedding_dimension // 2

    frequencies = torch.arange(half, dtype=torch.float32, device=x.device)
    frequencies = base ** -(frequencies / half)

    cos = torch.cos(x[:, None] * frequencies[None])
    sin = torch.sin(x[:, None] * frequencies[None])

    return torch.cat([cos, sin], dim=-1)


def frequency_encode_2d(
    x: torch.Tensor,
    embedding_dimension: int,
    base: float = 1e4,
) -> torch.Tensor:
    """Frequency encode a 2D index tensor."""

    assert embedding_dimension % 4 == 0, 'embedding_dimension must be divisible by 4.'

    quarter = embedding_dimension // 4

    frequencies = torch.arange(quarter, dtype=torch.float32, device=x.device)
    frequencies = base ** -(frequencies / quarter)

    cos_x = torch.cos(x[0, :, None] * frequencies[None])
    sin_x = torch.sin(x[0, :, None] * frequencies[None])
    cos_y = torch.cos(x[1, :, None] * frequencies[None])
    sin_y = torch.sin(x[1, :, None] * frequencies[None])

    return torch.cat([cos_x, sin_x, cos_y, sin_y], dim=-1)
