class ConvolutionalAttention(nn.Module):
    """Convolutional attention."""
    
    def __init__(
        self,
        *,
        in_channels: int,
    ) -> None:
        """Initialize the module."""
        
        super().__init__()

        self.project_q = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
        self.project_k = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
        self.project_v = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
        self.project_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        
        b, c, h, w = x.shape
        z = x
        
        q = self.project_q(z) 
        k = self.project_k(z)
        v = self.project_v(z)
        
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b (h w) c')
        
        z = F.softmax(q @ k, dim=-1) @ v
        z = rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
        
        return x + z
