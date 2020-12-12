import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        # query, key, value: [batch size, seq len, hid dim]
        # mask: [batch size, 1, 1, seq len]

        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)

        batch_size = query.size(0)
        
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # q,k,v: [batch size, num heads, seq len, head dim]

        energy = torch.matmul(q, k.permute(0,1,3,2)) / self.scale
        # energy: [batch size, num heads, seq len, seq len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        # attention: [batch size, num heads, seq len, seq len]

        x = torch.matmul(self.dropout(attention), v)
        x = x.permute(0,2,1,3).contiguous()
        # x: [batch size, seq len, num heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    # 两层全联接
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input: [batch size, src len, hid dim]
        output = torch.relu(self.fc1(input))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output


class EncoderLayer(nn.Module):
    # Encoder单层，self attention 全联接 都用残差连接
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.positionwise_fc = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src, src_mask):
        # src: [batch size, src len, hid dim]
        # src_mask: [batch size, 1, 1, src len]

        _src, _ = self.self_attention(src, src, src, src_mask)
        # residual & dropout
        src = self.layer_norm(src + self.dropout(_src))

        _src = self.positionwise_fc(src)
        src = self.layer_norm(src + self.dropout(_src))

        return src


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=310):
        super().__init__()

        self.device = device
        self.token_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)
        ])

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src: [batch size, src len]
        # src_mask: [batch size, 1, 1, src len]

        batch_size = src.size(0)
        src_len = src.size(1)

        pos_indices = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.token_embedding(src) * self.scale) + self.pos_embedding(pos_indices))

        for layer in self.layers:
            src = layer(src, src_mask)
        
        return src


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.positionwise_fc = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: [batch_size, trg_len, hid_dim]
        # enc_src: [batch_size, src_len, hid_dim]
        # trg_mask: [batch_size, 1, trg_len, trg_len]
        # src_mask: [batch_size, 1, 1, src_len]

        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.layer_norm(trg + self.dropout(_trg))

        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.layer_norm(trg + self.dropout(_trg))

        _trg = self.positionwise_fc(trg)
        trg = self.layer_norm(trg + self.dropout(_trg))

        return trg, attention


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=310):
        super().__init__()

        self.device = device
        self.token_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)
        ])

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sacle = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)


    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: [batch size, trg len]
        # enc_src: [batch size, src len, hid dim]
        # trg_mask: [batch size, 1, trg len, trg len]
        # src_mask: [batch size, 1, 1, src len]

        batch_size = trg.size(0)
        trg_len = trg.size(1)

        pos_indices = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        trg = self.dropout((self.token_embedding(trg) * self.sacle) + self.pos_embedding(pos_indices))

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        # trg: [batch size, trg len, hid_dim]
        # attention: [batch size, n_heads, trg_len, src_len]

        output = self.fc_out(trg)

        return output, attention


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def create_src_mask(self, src):
        # src: [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask: [batch size, 1, 1, src_len]
        return src_mask

    def create_trg_mask(self, trg):
        # trg: [batch_size, trg_len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        # trg_pad_mask: [batch_size, 1, trg_len, 1]
        trg_len = trg.size(1)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # before torch 1.2.0 version:
        # trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.uint8, device=self.device))

        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask: [batch_size, 1, trg_len, trg_len]
        return trg_mask


    def forward(self, src, trg):
        # src: [batch size, src len]
        # trg: [batch size, trg len]
        src_mask = self.create_src_mask(src)
        trg_mask = self.create_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output: [batch size, trg len, output dim]
        # attention: [batch size, num heads, trg_len, src_len]
        return output, attention


def build_model(input_dim, output_dim, device, pad_idx, trg_pad_idx,
    hid_dim=1024, enc_layers=2, dec_layers=2, enc_heads=4, dec_heads=4, enc_pf_dim=512, dec_pf_dim=512, enc_dropout=0.6, dec_dropout=0.6):

    encoder = Encoder(input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, device)
    decoder = Decoder(output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, device)

    model = Seq2seq(encoder, decoder, pad_idx, trg_pad_idx, device)
    model = model.to(device)
    return model

