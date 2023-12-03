"""Attention with relative position bias. Simliar to T5, however without chunking."""

class AttentionWithRelativePositionBias(nn.Module):

    def __init__(self, embedding_dimension: int, number_of_heads: int, maximum_sequence_length: int) -> None:

        super().__init__()

        self.number_of_heads = number_of_heads
        self.linear_qkv = nn.Linear(in_features=embedding_dimension, out_features=embedding_dimension * 3, bias=False)
        self.linear_out = nn.Linear(in_features=embedding_dimension, out_features=embedding_dimension, bias=False)
        self.relative_position_bias = nn.Parameter(1e-3 * torch.randn((number_of_heads, maximum_sequence_length, maximum_sequence_length)))  # One relative position matrix per head.

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        q, k, v = rearrange(self.linear_qkv(x), 'b s (k h e) -> k b h s e', k=3, h=self.number_of_heads)

        sequence_length = q.size(-2)
        bias = self.relative_position_bias[:, : sequence_length, : sequence_length]

        score = torch.einsum('bsqe,bske->bsqk', q, k)
        score = F.softmax((score + bias)/math.sqrt(k.size(1)) + mask, dim=-1)

        x = self.linear_out(rearrange(score @ v, 'b h s e -> b s (h e)'))

        return x
