import math
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

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)

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
        self.softmax = nn.Softmax(dim=3)

    def forward(self, q, k, v, mask=None, e=1e-12):
        """q, k, v: [batch_size, head, length, d_model/n_head]
        """
        batch_size, head, length, d_tensor = k.size()

        k_t = k.view(batch_size, head, d_tensor, length)
        # batched matrix multiply
        score = (q @ k_t) / math.sqrt(d_tensor)

        if mask is not None:
            # fills element of tensor where mask is True with -e
            # mask is either batch_size × 1 × 1 × length (src_mask)
            # or batch_size × 1 × length × length (trg_mask, masked mha)
            score = score.masked_fill(mask == 0, -e)
        
        score = self.softmax(score)

        v = score @ v

        return v, score

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        # added to the list of parameters
        # can't find these scaling parameters in the transformer paper
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
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
        
class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(
            d_model=d_model,
            max_len=max_len,
            vocab_size=enc_voc_size,
            drop_prob=drop_prob,
            device=device)
        
        self.layers = nn.ModuleList([EncoderLayer(
            d_model,
            ffn_hidden,
            n_head,
            drop_prob)
            for _ in range(n_layers)])
    
    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            x = self.dropout2(x)
            x = self.norm2(x + _x)
        
        _x = x
        x = self.ffn(x)

        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(
            dec_voc_size,
            d_model,
            max_len,
            drop_prob,
            device
        )

        self.layers = nn.ModuleList([DecoderLayer(
            d_model,
            ffn_hidden,
            n_head,
            drop_prob
        ) for i in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
        
        output = self.linear(trg)
        return output

class Transformer(nn.Module):
    # sos: start of stream, not used by this implementation…
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len, ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        # dim: batch_size × 1 × seq_len × 1
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        # dim: seq_len × seq_len
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.BoolTensor).to(self.device)
        # dim: batch_size × 1 × seq_len × seq_len
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask