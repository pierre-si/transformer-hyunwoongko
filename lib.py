import torch
import torch.nn as nn

# fixed positional encodings
# same dimension as the embeddings so that they can be summed
# The wavelengths form a geometric progression from 2π to 10000 · 2π. Why 10k×2π?
class PositionalEncoding(nn.Module):
    """sine and cosine
    """
    def __init__(self, d_model, max_len, device):
        super().__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1) # [max_len×1]

        _2i = torch.arange(0, d_model, step=2, device=device).float() # [d_model/2] (//2+1)
        self.encoding[:, 0::2] = torch.sin(pos / 10_000**(_2i / d_model)) # [max_len × d_model/2]
        self.encoding[:, 1::2] = torch.cos(pos / 10_000**(_2i / d_model))

    # apparemment x est le vecteur de token et non l'embedding
    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]
