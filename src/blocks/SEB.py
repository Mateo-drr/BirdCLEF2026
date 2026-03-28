
from torch import nn

class SEBlock(nn.Module):
    """ Squeeze and Excite block"""
    def __init__(self, channels: int, reduce_dim: int=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(nn.Linear(channels, reduce_dim, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(reduce_dim, channels, bias=False),
                                    nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c) # Squeeze: [b,c,h,w] → [b,c,1,1] → [b,c]
        y = self.excite(y) # Excite: [b,c] → [b,r] → [b,c]
        return x * y.view(b, c, 1, 1) # Scale: [b,c] → [b,c,1,1] then multiply

