# Note: this is a simplified ALiBi where the slope is 1 for all heads.

class AttentionWithLinearBias(nn.Module):

    def __init__(self, embedding_dimension: int, number_of_heads: int, maximum_sequence_length: int) -> None:

        super().__init__()

        self.number_of_heads = number_of_heads
        self.linear_qkv = nn.Linear(in_features=embedding_dimension, out_features=embedding_dimension * 3, bias=False)
        self.linear_out = nn.Linear(in_features=embedding_dimension, out_features=embedding_dimension, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        q, k, v = rearrange(self.linear_qkv(x), 'b s (k h e) -> k b h s e', k=3, h=self.number_of_heads)

        sequence_length = q.size(-2)
        bias = torch.tril(torch.arange(sequence_length) - torch.arange(sequence_length).unsqueeze(1))

        score = torch.einsum('bsqe,bske->bsqk', q, k)
        score = F.softmax((score + bias)/math.sqrt(k.size(1)) + mask, dim=-1)

        x = self.linear_out(rearrange(score @ v, 'b h s e -> b s (h e)'))

        return x
