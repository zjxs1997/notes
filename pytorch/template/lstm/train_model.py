# 新写的orz

import math
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim

from hparam import get_parser
from load_dataset import build_dataset
from model import build_model


parser = get_parser()
hparam = parser.parse_args()

if hparam.device is not None:
    device = torch.device(hparam.device)
else:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, iterator, optimizer, criterion, hparam,):
    clip = hparam.clip_grad_norm
    print_every = hparam.train_print_loss_every

    model.train()

    total_epoch_loss = 0
    
    for bi, batch in enumerate(iterator):
        src, src_len = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, src_len, trg)
        # [trg len - 1, batch size, output dim]
        output_dim = output.size(2)

        output = output.view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        total_epoch_loss += loss.item()

        if (bi + 1) % print_every == 0:
            print('\r %d | %f' % (bi + 1, total_epoch_loss / (bi + 1)), end='  ')
    print('\n')
    return total_epoch_loss


def eval(model, iterator, criterion,):
    model.eval()
    total_epoch_loss = 0

    with torch.no_grad():
        for bi, batch in enumerate(iterator):
            src, src_len = batch.src
            trg = batch.trg

            output = model(src, src_len, trg, teacher_forcing_rate=0)
            output_dim = output.size(2)
            output = output.view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            total_epoch_loss += loss.item()

    return total_epoch_loss


def time_convert(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_seconds = int(elapsed_time - elapsed_mins * 60)
    return elapsed_mins, elapsed_seconds


if __name__ == "__main__":
    src_field, trg_field, train_iterator, val_iterator = build_dataset(hparam)

    save_path = hparam.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    pickle.dump(src_field, open(f'{save_path}/source_field.pt', 'wb'))
    pickle.dump(trg_field, open(f'{save_path}/target_field.pt', 'wb'))

    model = build_model(hparam, src_field, trg_field).to(device)
    model.apply(init_weights)
    print(count_parameters(model))

    optimizer = optim.Adam(model.parameters())
    trg_pad_idx = trg_field.vocab.stoi[trg_field.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    best_val_loss = float("inf")
    for ei in range(hparam.train_epoch):
        print("epoch " + str(ei))
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, hparam)
        train_loss = train_loss / len(train_iterator)
        torch.save(model.state_dict(), f'{save_path}/checkpoint{ei}.pt')
        print(f"train loss: {train_loss:.6f}")

        val_loss = eval(model, val_iterator, criterion)
        val_loss = val_loss / len(val_iterator)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{save_path}/best.pt')
        print(f"val loss: {val_loss:.6f} || best val loss: {best_val_loss:.6f}")

        end_time = time.time()
        print("=" * 60, end = '')
        print(f"cost time: {end_time - start_time:.2f} seconds")
