import math
import pickle
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from load_dataset import build_dataset
from model import build_model
from model_decode import decode_path

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

NUM_EPOCH = 200

argv = sys.argv
suffix = '' if len(argv) == 1 else argv[1]


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step=0):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def zero_grad(self):
        self.optimizer.zero_grad()



def init_weight(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_normal_(m.weight.data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_convert(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_seconds = int(elapsed_time - elapsed_mins * 60)
    return elapsed_mins, elapsed_seconds


def train(model, iterator, optimizer, criterion, clip=1, print_every=100):
    model.train()

    total_epoch_loss = 0

    for bi, batch in enumerate(iterator):
        src = batch.passage
        trg = batch.question

        src = src.to(device)
        trg = trg.to(device)
        # src: [batch size, src len]
        # trg: [batch size, trg len]

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])
        # output: [batch size, trg len - 1, output dim]
        output_dim = output.size(2)
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

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
            src = batch.passage
            trg = batch.question

            src = src.to(device)
            trg = trg.to(device)

            output, _ = model(src, trg[:, :-1])
            output_dim = output.size(2)
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            total_epoch_loss += loss.item()

    return total_epoch_loss / len(iterator)


if __name__ == "__main__":
    train_iterator, val_iterator, test_iterator, QUESTION, TITLE, PASSAGE, BOOL = build_dataset()
    model = build_model(len(PASSAGE.vocab), len(QUESTION.vocab), device, PASSAGE.vocab.stoi[PASSAGE.pad_token], QUESTION.vocab.stoi[QUESTION.pad_token])

    pickle.dump(easy_field(QUESTION), open('question_field.pt', 'wb'))
    pickle.dump(easy_field(PASSAGE), open('passage_field.pt', 'wb'))

    # pickle.dump(easy_field(BOOL), open('bool_field.pt', 'wb'))

    # model.apply(init_weight)
    model.load_state_dict(torch.load(f"checkpoint_{suffix}.pt"))

    print(count_parameters(model))

    optimizer = optim.Adam(model.parameters(), lr=0)
    optimizer = NoamOpt(2048, 2, 16000, optimizer, step=0)
    PAD_IDX = QUESTION.vocab.stoi[QUESTION.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_valid_loss = float("inf")
    best_valid_epoch = -1

    for ei in range(NUM_EPOCH):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion)
        val_loss = eval(model, val_iterator, criterion)

        end_time = time.time()
        elapsed_mins, elapsed_seconds = time_convert(start_time, end_time)

        torch.save(model.state_dict(), f'checkpoint_{suffix}.pt')

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_valid_epoch = ei + 1
            torch.save(model.state_dict(), f'best_model_{suffix}.pt')
        
        print(f'Epoch: {ei+1:02} | Time: {elapsed_mins}m {elapsed_seconds}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f} | best val loss: {best_valid_loss:.3f} | best val epoch: {best_valid_epoch}')


    # model.load_state_dict(torch.load(f"best_model_{suffix}.pt"))
    test_loss = eval(model, test_iterator, criterion)
    print(f'\t Test. Loss: {test_loss:.3f} |  Test. PPL: {math.exp(test_loss):7.3f}')

    cb, decode_pred = decode_path(model, '../data/test.tsv', PASSAGE, QUESTION)
    print(cb, end=' | ')
    pickle.dump(decode_pred, open(f'test_pred_{suffix}.pt', 'wb'))
    print(f'saved to test_pred_{suffix}.pt')


