class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, embedding_dimension: int, base: int = 10_000) -> None:

        super().__init__()

        assert embedding_dimension % 2 == 0

        theta = base ** (-2 * torch.arange(embedding_dimension//2) / embedding_dimension).cuda()
        theta = theta.repeat_interleave(2)

        self.theta = theta
        self.alpha = (-1) ** (torch.arange(embedding_dimension) + 1).cuda()  # Alternating -1, 1, -1, ...
    
    def rotate(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:

        cos = torch.cos(self.theta * position.unsqueeze(1))
        sin = torch.sin(self.theta * position.unsqueeze(1))

        return (x * cos) + (self.alpha * x * sin)


class AttentionWithRotaryPositionalEmbedding(nn.Module):

    def __init__(self, embedding_dimension: int, number_of_heads: int, maximum_sequence_length: int) -> None:

        super().__init__()

        self.number_of_heads = number_of_heads
        self.linear_qkv = nn.Linear(in_features=embedding_dimension, out_features=embedding_dimension * 3, bias=False)
        self.linear_out = nn.Linear(in_features=embedding_dimension, out_features=embedding_dimension, bias=False)
        self.rotary = RotaryPositionalEmbedding(embedding_dimension=embedding_dimension//number_of_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        q, k, v = rearrange(self.linear_qkv(x), 'b s (k h e) -> k b h s e', k=3, h=self.number_of_heads)
        position = torch.arange(q.size(-2)).cuda()

        q = self.rotary.rotate(q, position)
        k = self.rotary.rotate(k, position)

        score = torch.einsum('bsqe,bske->bsqk', q, k)
        score = F.softmax(score/math.sqrt(k.size(1)) + mask, dim=-1)

        x = self.linear_out(rearrange(score @ v, 'b h s e -> b s (h e)'))

        return x
