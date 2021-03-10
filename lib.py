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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        # the projections for the different heads are done with the same matrix
        # they are then split and the attention is applied
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # matrix for the final projection after concaneting the attention outputs
        self.w_concat = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        out, attention = self.attention(q, k, v, mask=mask)

        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """split along d_model by number of heads 
        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, n_head, length, d_model/n_head]
        """

        batch_size, length, d_model = tensor.size()
        d_tensor =  d_model // self.n_head

        tensor = tensor.view(batch_size, self.n_head, length, d_tensor)

        return tensor

    def concat(self, tensor):
        """inverse function of self.split(tensor: torch.Tensor)

        :param tensor: [batch_size, n_head, length, d_model/n_head]
        :return: [batch_size, length, d_model]
        """

        batch_size, n_head, length, d_tensor = tensor.size()
        d_model = n_head * d_tensor

        return tensor.view(batch_size, length, d_model)

class ScaleDotProductAttention(nn.Module):
    """computes scale dot product attention
    Query: vector to compare to every other vector to establish the weights for its own output
    Key: vector to compare to the query to establish the weights for the query's output
    Value: vector used as part of the weighted sum to compute the output vector
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None, e=1e-12):
        """q, k, v: [batch_size, head, length, d_model/n_head]
        """
        batch_size, head, length, d_tensor = q.size()

        k_t = k.view(batch_size, head, d_tensor, length)
        # batched matrix multiply
        score = (q @ k_t) / math.sqrt(d_tensor)

        if mask is not None:
            # fills element of tensor where mask is True with -e
            score = score.masked_fill(mask == 0, -e)
        
        score = self.softmax(score)

        v = score @ v

        return v, score

class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # added to the list of parameters
        # can't find these scaling parameters in the transformer paper
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(x):
        # -1: on the last dimension only
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out  

# position wise : applied to each position (length dim) separately and identically.
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden=2048, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, src_mask):
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # dropout before res and norm as per the paper
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
        