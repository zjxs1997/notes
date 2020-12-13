# 改完了

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder_pack(nn.Module):
    """
    input:
        src: [src len, batch size]
        src_len: [batch size]
    output:
        outputs: [src len, batch size, enc hid dim * direction]
        hidden: [batch size, dec hid dim]
        cell: [batch size, dec hid dim]
    """
    def __init__(self, hparam, source_field):
        super().__init__()
        input_dim = len(source_field.vocab.itos)
        emb_dim = hparam.enc_emb_dim
        enc_hid_dim = hparam.enc_hid_dim
        dropout = hparam.enc_dropout
        num_layers = hparam.enc_num_layer

        dec_hid_dim = hparam.dec_hid_dim


        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.cell_fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len,):
        # src: [src len, batch size]
        # src_len: [batch size]

        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        # [src len, batch size, emb dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)

        # packed_outputs, hidden = self.gru(packed_embedded)
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # hidden: [num layer * num direction, batch size, enc hid dim]
        # hidden[-2, :, :] 是rnn最后一层的前向输出
        # hidden[-1, :, :] 是rnn最后一层的反向输出
        # outputs: [src len, batch size, enc hid dim * num direction]，包含了整个序列的encode信息

        # hidden = torch.cat(
        #     (hidden[::2, :, :], hidden[1::2, :, :]),
        #     dim = 2
        # )
        hidden = torch.cat(
            (hidden[-2,:,:], hidden[-1,:,:]),
            dim=1
        )
        hidden = self.fc(hidden)
        hidden = torch.tanh(hidden)
        # hidden: [batch size, dec hid dim]，只包含rnn encoder最终的状态，且作为decoder的初始输入
        # 考虑怎么把hidden变成：[num layers, batch size, dec hid dim]来让decoder也用多层；
        # 不用考虑了，decoder多层效果更差

        cell = torch.cat(
            (cell[-2,:,:], cell[-1,:,:]),
            dim=1
        )
        cell = self.fc(cell)
        cell = torch.tanh(cell)

        return outputs, hidden, cell


class Attention(nn.Module):
    # 输入解码状态hidden和编码器最后的编码矩阵，返回attention值
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch size, dec hid dim]
        # encoder_outputs: [src len, batch size, enc hid dim * num direction]
        # mask: [batch size, src len] 0 & 1
        # return, attention: [batch size, src len]
        
        batch_size = encoder_outputs.size(1)
        src_len = encoder_outputs.size(0)

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden: [batch size, src len, dec hid dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs: [batch size, src len, enc hid dim * num direction]

        energy = torch.cat(
            (hidden, encoder_outputs),
            dim = 2
        )
        energy = torch.tanh(self.attn(energy))
        # [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        # [batch size, src len]
        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    # 解码单个时序，输入上个时序的解码结果等，输出预测、隐向量、attention矩阵
    def __init__(self, target_field, hparam, attention):
        super().__init__()
        self.attention = attention

        output_dim = len(target_field.vocab.itos)
        emb_dim = hparam.dec_emb_dim
        enc_hid_dim = hparam.enc_hid_dim
        dec_hid_dim = hparam.dec_hid_dim
        dropout = hparam.dec_dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim, bidirectional=False)
        self.fc = nn.Linear((enc_hid_dim * 2) + emb_dim + dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs, mask,):
        # input: [batch size]
        # hidden & cell: [batch size, dec hid dim]
        # encoder_outputs: [src len, batch size, enc hid dim * num direction=2]
        # mask: [batch size, src len]
        # return: prediction: [batch size, output dim]
        # hidden & cell: [batch size, dec hid dim]
        # attention: [batch size src len]

        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        # [1, batch size, emb dim]

        attention = self.attention(hidden, encoder_outputs, mask)
        attention = attention.unsqueeze(1)
        # [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # [batch size, src len, enc hid dim * num direction]
        weighted_vector = torch.bmm(attention, encoder_outputs)
        # [batch size, 1, enc hid dim * num direction]
        weighted_vector = weighted_vector.permute(1, 0, 2)
        # [1, batch size, enc hid dim * num direction]

        lstm_input = torch.cat(
            (weighted_vector, embedded),
            dim = 2
        )
        output, (hidden, cell) = self.lstm(lstm_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        # output: [seq len = 1, batch size, dec hid dim]
        # hidden: [num layer * num direction = 1, batch size, dec hid dim]

        pred_input = torch.cat(
            (output, weighted_vector, embedded),
            dim = 2
        )
        pred_input = pred_input.squeeze(0)
        prediction = self.fc(pred_input)
        # [batch size, output dim]

        return prediction, hidden.squeeze(0), cell.squeeze(0), attention.squeeze(1)


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, source_field,):
        super().__init__()
        source_vocab = source_field.vocab
        self.encoder = encoder
        self.decoder = decoder
        self.source_pad_idx = source_vocab.stoi[source_field.pad_token]
    
    def create_mask(self, data, pad_idx=None):
        if pad_idx is None:
            pad_idx = self.source_pad_idx
        # data: [seq len, batch size]
        mask = (data != pad_idx).permute(1, 0)
        # mask: [batch size, seq len]
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_rate=0.5):
        # src: [src len, batch size]
        # src_len: [batch size]
        # trg: [trg len, batch size]
        # return, outputs: [trg len -1, batch size, output dim]

        trg_len = trg.size(0)
        outputs = []
        encoder_outputs, hidden, cell = self.encoder(src, src_len,)

        input = trg[0, :]
        # <sos>
        mask = self.create_mask(src)

        for ti in range(1, trg_len):
            output, hidden, cell, attention = self.decoder(input, hidden, cell, encoder_outputs, mask,)
            # output: [batch size, output dim]
            outputs.append(output)
            teacher_force = random.random() < teacher_forcing_rate
            top1 = output.argmax(1)
            # [batch size]
            input = trg[ti] if teacher_force else top1
        
        outputs = torch.stack(outputs, dim=0)
        # [trg len - 1, batch size, trg vocab size]
        return outputs


def build_model(hparam, source_field, target_field):
    attention = Attention(hparam.enc_hid_dim, hparam.dec_hid_dim)
    encoder = Encoder_pack(hparam, source_field)
    decoder = Decoder(target_field, hparam, attention)

    model = Seq2seq(encoder, decoder, source_field)
    return model


