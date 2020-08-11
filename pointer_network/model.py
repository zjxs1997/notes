import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, num_layers=2, dropout=dropout)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src: [src len, batch size]
        # src_len: [src len]

        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        # [src len, batch size, emb dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)

        packed_outputs, hidden = self.gru(packed_embedded)
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
            (hidden[::2,:,:], hidden[1::2,:,:]),
            dim=2
        )

        hidden = self.fc(hidden)
        hidden = torch.tanh(hidden)
        # hidden: [batch size, dec hid dim]，只包含rnn encoder最终的状态，且作为decoder的初始输入
        # 考虑怎么把hidden变成：[num layers, batch size, dec hid dim]

        return outputs, hidden


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
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, pointer_attention):
        super().__init__()
        self.attention = attention
        self.pointer_attention = pointer_attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, bidirectional=False, num_layers=2, dropout=dropout)
        self.fc = nn.Linear((enc_hid_dim * 2) + emb_dim + dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.p_gen = nn.Linear(enc_hid_dim*2+dec_hid_dim, 1)

    def forward(self, src, input, hidden, encoder_outputs, mask):
        # input: [batch size]
        # hidden: [num layer, batch size, dec hid dim]
        # encoder_outputs: [src len, batch size, enc hid dim * num direction]
        # mask: [batch size, src len]

        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        # [1, batch size, emb dim]

        attention = self.attention(hidden[-1,:,:], encoder_outputs, mask)
        attention = attention.unsqueeze(1)
        # [batch size, 1, src len]
        pointer_attention = self.pointer_attention(hidden[-1,:,:], encoder_outputs, mask)
        # [batch size, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # [batch size, src len, enc hid dim * num direction]

        weighted_vector = torch.bmm(attention, encoder_outputs)
        # [batch size, 1, enc hid dim * num direction]
        weighted_vector = weighted_vector.permute(1, 0, 2)
        # [1, batch size, enc hid dim * num direction]

        # print(weighted_vector.shape)
        # print(hidden.shape)
        p_gen_input = torch.cat((weighted_vector.squeeze(0), hidden[-1,:,:]), dim=1)
        p_gen = torch.sigmoid(self.p_gen(p_gen_input)).squeeze(1)
        # [batch size]

        gru_input = torch.cat(
            (weighted_vector, embedded),
            dim = 2
        )
        output, hidden = self.gru(gru_input, hidden)
        # output: [seq len = 1, batch size, dec hid dim]
        # hidden: [num layer * num direction = 1, batch size, dec hid dim]

        pred_input = torch.cat(
            (output, weighted_vector, embedded),
            dim = 2
        )
        pred_input = pred_input.squeeze(0)
        prediction = self.fc(pred_input)
        # [batch size, output dim]

        pointer_prediction = torch.zeros_like(prediction)
        pointer_prediction = pointer_prediction.scatter_add(1, src.permute(1, 0), pointer_attention)
        pointer_prediction = pointer_prediction * p_gen.unsqueeze(1)

        prediction = prediction * (1-p_gen).unsqueeze(1)
        prediction = prediction + pointer_prediction

        return prediction, hidden, attention.squeeze(1)


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx

    def create_mask(self, src):
        # src: [src len, batch size]
        mask = (src != self.src_pad_idx).permute(1, 0)
        # mask: [batch size, src len]
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_rate=0.5):
        # src: [src len, batch size]
        # src_len: [batch size]，src这个batch中每个序列的真实长度
        # trg: [trg len, batch size]

        trg_len = trg.size(0)
        outputs = []

        encoder_outputs, hidden = self.encoder(src, src_len)

        input = trg[0, :]
        # <sos>
        mask = self.create_mask(src)

        for ti in range(1, trg_len):
            output, hidden, _ = self.decoder(src, input, hidden, encoder_outputs, mask)
            # output: [batch size, output dim]
            outputs.append(output)
            teacher_force = random.random() < teacher_forcing_rate
            top1 = output.argmax(1)
            # [batch size]
            input = trg[ti] if teacher_force else top1
        
        outputs = torch.stack(outputs, dim=0)
        # [trg len - 1, batch size, trg vocab size]
        return outputs


def build_model(input_dim, output_dim, src_pad_idx, device,
    enc_emb_dim=512, dec_emb_dim=512, enc_hid_dim=512, dec_hid_dim=512, enc_dropout=0.5, dec_dropout=0.5):

    attention = Attention(enc_hid_dim, dec_hid_dim)
    pointer_attention = Attention(enc_hid_dim, dec_hid_dim)
    encoder = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
    decoder = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attention, pointer_attention)

    model = Seq2seq(encoder, decoder, src_pad_idx)
    model = model.to(device)
    
    return model







