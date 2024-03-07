def frequency_encode(
    x: torch.Tensor,
    embedding_dimension: int,
    base: float = 1e4,
) -> torch.Tensor:
    """Freqency encode an index tensor."""

    half = embedding_dimension // 2

    frequencies = torch.arange(half, dtype=torch.float32, device=x.device)
    frequencies = base ** -(frequencies / half)

    sin = torch.sin(x[:, None] * frequencies[None])
    cos = torch.cos(x[:, None] * frequencies[None])

    return torch.cat([cos, sin], dim=-1)
