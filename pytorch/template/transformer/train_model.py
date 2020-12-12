# 大概也没问题了吧orz

import pickle
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim

from hparam import get_parser
from load_dataset import build_dataset
from model import build_model


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, hparam, optimizer, step=0):
        self.optimizer = optimizer
        self._step = step
        self.warmup = hparam.warmup_steps
        self.factor = hparam.optim_factor
        self.model_size = hparam.hid_dim
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

arg_parser = get_parser()
hparam = arg_parser.parse_args()

def train_model(model, iterator, optimizer, criterion, hparam,):
    model.train()

    start_time = time.time()
    total_epoch_loss = 0
    optimizer.zero_grad()
    accumulate_loss = 0

    for bi, batch in enumerate(iterator):
        src = batch.src
        trg, trg_lens = batch.trg

        output, attention = model(src, trg[:, :-1],)
        vocab_size = output.size(2)
        output = output.contiguous().view(-1, vocab_size)
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        accumulate_loss += loss / hparam.gradient_accumulate
        total_epoch_loss += loss.item()

        if (bi+1) % hparam.gradient_accumulate == 0:
            accumulate_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hparam.clip)
            optimizer.step()
            optimizer.zero_grad()
            accumulate_loss = 0
        
        if (bi+1) % hparam.print_loss_every_batch == 0:
            current_time = time.time()
            time_elapsed = current_time - start_time
            print(f"\r batch {bi+1} | total_epoch_loss: {total_epoch_loss / (bi+1)} | time: {time_elapsed:.2f}", end='  ')
    if (bi+1) % hparam.gradient_accumulate != 0:
        accumulate_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparam.clip)
        optimizer.step()
    print('\n')

    return total_epoch_loss

def eval_model(model, iterator, criterion,):
    model.eval()
    total_epoch_loss = 0
    with torch.no_grad():
        for bi, batch in enumerate(iterator):
            src = batch.src
            trg, trg_lens = batch.trg

            output, attention = model(src, trg[:, :-1],)
            vocab_size = output.size(2)
            output = output.contiguous().view(-1, vocab_size)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            total_epoch_loss += loss.item()
    return total_epoch_loss


if hparam.device is None:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device(hparam.device)

try:
    if __name__ == "__main__":
        save_path = hparam.save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    
        source_field, target_field, train_iterator, valid_iterator = build_dataset(hparam)

        target_pad_idx = target_field.vocab.stoi[target_field.pad_token]

        pickle.dump(source_field, open(f"{save_path}/source_field.pkl", 'wb'))
        pickle.dump(target_field, open(f"{save_path}/target_field.pkl", 'wb'))

        model = build_model(hparam, source_field, target_field, device)
        optimizer = optim.Adam(model.parameters(), lr=hparam.model_learning_rate)
        optimizer = NoamOpt(hparam, optimizer,)
        criterion = nn.CrossEntropyLoss(ignore_index=target_pad_idx)

        if hparam.load_model is not None and os.path.exists(hparam.load_model):
            model.load_state_dict(torch.load(hparam.load_model))

        best_val_loss = float('inf')

        for ei in range(hparam.train_epoch):
            print(f"start training epoch {ei+1}:")
            train_loss = train_model(model, train_iterator, optimizer, criterion, hparam)
            train_loss = train_loss / (len(train_iterator))
            print(f'train loss: {train_loss:.4f}')

            val_loss = eval_model(model, valid_iterator, criterion)
            val_loss = val_loss / len(valid_iterator)
            print(f'val loss: {val_loss:.4f}')

            torch.save(model.state_dict(), f'{save_path}/checkpoint{ei}.pt')

            if val_loss < best_val_loss:
                val_loss = best_val_loss
                torch.save(model.state_dict(), f'{save_path}/best.pt')


except Exception as e:
    print(e)
    __import__('ipdb').post_mortem()
