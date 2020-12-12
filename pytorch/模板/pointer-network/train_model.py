import math
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim

from load_dataset import build_dataset
from model import build_model
from util import easy_field

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

NUM_EPOCH = 100

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, iterator, optimizer, criterion, clip=1, print_every=100):
    model.train()

    total_epoch_loss = 0
    
    for bi, batch in enumerate(iterator):
        src, src_len = batch.passage
        trg, trg_len = batch.question

        optimizer.zero_grad()

        src = src.to(device)
        src_len = src_len.to(device)
        trg = trg.to(device)

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


    return total_epoch_loss / len(iterator)


def eval(model, iterator, criterion):
    model.eval()
    total_epoch_loss = 0

    with torch.no_grad():
        for bi, batch in enumerate(iterator):
            src, src_len = batch.passage
            trg, trg_len = batch.question

            src = src.to(device)
            src_len = src_len.to(device)
            trg = trg.to(device)

            output = model(src, src_len, trg, 0)
            output_dim = output.size(2)
            output = output.view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            total_epoch_loss += loss.item()

    return total_epoch_loss / len(iterator)


def time_convert(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_seconds = int(elapsed_time - elapsed_mins * 60)
    return elapsed_mins, elapsed_seconds




if __name__ == "__main__":
    train_iterator, val_iterator, test_iterator, field, bool_field = build_dataset()
    model = build_model(len(field.vocab), len(field.vocab), field.vocab.stoi[field.pad_token], device)

    pickle.dump(easy_field(field), open("word_field.pt", 'wb'))

    # model.apply(init_weights)
    model.load_state_dict(torch.load("checkpointl.pt"))
    print(count_parameters(model))

    optimizer = optim.Adam(model.parameters())
    TRG_PAD_IDX = field.vocab.stoi[field.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    best_valid_loss = float('inf')

    try:
        for ei in range(NUM_EPOCH):
            start_time = time.time()

            train_loss = train(model, train_iterator, optimizer, criterion)
            val_loss = eval(model, val_iterator, criterion)
            
            end_time = time.time()
            elapsed_mins, elapsed_seconds = time_convert(start_time, end_time)

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                torch.save(model.state_dict(), 'best_modell.pt')
            torch.save(model.state_dict(), 'checkpointl.pt')
            
            print(f'Epoch: {ei+1:02} | Time: {elapsed_mins}m {elapsed_seconds}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')
    except KeyboardInterrupt as e:
        print("kkkk")


    model.load_state_dict(torch.load("best_modell.pt"))
    test_loss = eval(model, test_iterator, criterion)
    print(f'\t Test. Loss: {test_loss:.3f} |  Test. PPL: {math.exp(test_loss):7.3f}')


