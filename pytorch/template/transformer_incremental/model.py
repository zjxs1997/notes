# 应该没问题。。。

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

    def forward(self, query, key, value, mask=None, layer_cache=None, attn_type='self'):
        # query, key, value: [batch size, seq len, hid dim]
        # mask: [batch size, 1, 1, seq len]
        
        # if layer_cache is not None:
        # if includes key and value
        # and the length of query should be 1

        q = self.fc_q(query)
        batch_size = query.size(0)
        
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # q: [batch size, num heads, seq len, head dim]

        if layer_cache is not None:
            if attn_type == 'self':
                # q, k, v的seq len都是1
                k = self.fc_k(key)
                v = self.fc_v(value)
                k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
                v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

                if layer_cache['self_keys'] is not None:
                    k = torch.cat((layer_cache['self_keys'], k), dim=2)
                if layer_cache['self_values'] is not None:
                    v = torch.cat((layer_cache['self_values'], v), dim=2)
                layer_cache['self_keys'] = k
                layer_cache['self_values'] = v
            elif attn_type == 'context':
                if layer_cache['memory_keys'] is None:
                    k = self.fc_k(key)
                    v = self.fc_v(value)
                    k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
                    v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
                else:
                    k = layer_cache['memory_keys']
                    v = layer_cache['memory_values']
                layer_cache['memory_keys'] = k
                layer_cache['memory_values'] = v
            else:
                print("??? attention type error!")
                exit(0)
        else:
            # 没有layer_cache，多半是在训练，
            k = self.fc_k(key)
            v = self.fc_v(value)
            k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(q, k.permute(0,1,3,2)) / self.scale
        # energy: [batch size, num heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        # attention: [batch size, num heads, query len, key len]

        x = torch.matmul(self.dropout(attention), v)
        x = x.permute(0,2,1,3).contiguous()
        # x: [batch size, query len, num heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim)
        # x: [batch size, query len, hid dim]
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
    def __init__(self, hparam, source_field, device, ):
        super().__init__()
        source_vocab = source_field.vocab
        input_dim = len(source_vocab.itos)
        max_length = hparam.source_max_len
        dropout = hparam.encode_dropout
        hid_dim = hparam.hid_dim
        n_layers = hparam.enc_layers
        n_heads = hparam.enc_heads
        pf_dim = hparam.enc_pf_dim

        self.device = device
        self.token_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.dropout = nn.Dropout(dropout)

        self.pad_idx = source_vocab.stoi[source_field.pad_token]

        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)
        ])

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask=None):
        # src: [batch size, src len]
        # src_mask: [batch size, 1, 1, src len]
        if src_mask is None:
            src_mask = self.create_src_mask(src)

        batch_size = src.size(0)
        src_len = src.size(1)

        pos_indices = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout(
            (self.token_embedding(src) * self.scale) + 
            self.pos_embedding(pos_indices)
        )

        for layer in self.layers:
            src = layer(src, src_mask)
        
        return src
    
    def create_src_mask(self, data):
        src_mask = (data != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask



class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.positionwise_fc = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, trg, enc_src, trg_mask, src_mask, layer_cache=None, ):
        # trg: [batch_size, trg_len, hid_dim]
        # enc_src: [batch_size, src_len, hid_dim]
        # trg_mask: [batch_size, 1, trg_len, trg_len]
        # src_mask: [batch_size, 1, 1, src_len]

        _trg, _ = self.self_attention(trg, trg, trg, trg_mask, layer_cache=layer_cache, attn_type='self')
        trg = self.layer_norm(trg + self.dropout(_trg))

        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask, layer_cache=layer_cache, attn_type='context')
        trg = self.layer_norm(trg + self.dropout(_trg))

        _trg = self.positionwise_fc(trg)
        trg = self.layer_norm(trg + self.dropout(_trg))

        return trg, attention


class Decoder(nn.Module):
    def __init__(self, hparam, target_field, device,):
        super().__init__()

        target_vocab = target_field.vocab
        output_dim = len(target_vocab.itos)
        trg_pad_idx = target_vocab.stoi[target_field.pad_token]
        hid_dim = hparam.hid_dim
        n_layers = hparam.dec_layers
        n_heads = hparam.dec_heads
        pf_dim = hparam.dec_pf_dim
        dropout = hparam.decode_dropout
        max_length = hparam.target_max_len

        self.device = device
        self.trg_pad_idx = trg_pad_idx
        self.token_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)
        ])

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sacle = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def init_layer_caches(self):
        self.layer_caches = [{'self_keys': None, 'self_values': None, 'memory_keys': None, 'memory_values': None} for _ in self.layers]

    def forward(self, trg, enc_src, trg_mask, src_mask,):
        # trg: [batch size, trg len]
        # enc_src: [batch size, src len, hid dim]
        # trg_mask: [batch size, 1, trg len, trg len]
        # src_mask: [batch size, 1, 1, src len]

        batch_size = trg.size(0)
        trg_len = trg.size(1)

        pos_indices = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        trg = self.dropout(
            (self.token_embedding(trg) * self.sacle) + 
            self.pos_embedding(pos_indices)
        )

        if hasattr(self, 'layer_caches'):
            for layer, layer_cache in zip(self.layers, self.layer_caches):
                trg, attention = layer(trg, enc_src, trg_mask, src_mask, layer_cache=layer_cache)
        else:
            for layer in self.layers:
                trg, attention = layer(trg, enc_src, trg_mask, src_mask,)
            # trg: [batch size, trg len, hid_dim]
            # attention: [batch size, n_heads, trg_len, src_len]

        output = self.fc_out(trg)

        return output, attention


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


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg,):
        # src: [batch size, src len]
        # trg: [batch size, trg len]
        src_mask = self.encoder.create_src_mask(src)
        trg_mask = self.decoder.create_trg_mask(trg)

        # import ipdb; ipdb.set_trace()
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output: [batch size, trg len, output dim]
        # attention: [batch size, num heads, trg_len, src_len]
        return output, attention


def build_model(hparam, source_field, target_field, device):

    encoder = Encoder(hparam, source_field, device)
    decoder = Decoder(hparam, target_field, device)
    model = Seq2seq(encoder, decoder, device)
    model = model.to(device)

    return model

