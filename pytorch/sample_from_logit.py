import numpy as np
import torch
from torch.distributions import Categorical


# input: logit [batch size, vocab size]
# output: indices[batch size]
def top_p_sample(logits, p=0.9):
    # 如果要用带temperature的sample的话，在softmax之前让logits除以temperature即可。
    # temperature越小，会导致prob之间差距拉的越开，sample的效果更接近greedy
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = probs.sort(descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cum_probs > p
    sorted_indices_to_remove = sorted_indices_to_remove[..., :-1]

    # assume input is shaped (batch size, vocab dim)
    for bi in range(logits.size(0)):
        indices_to_remove = sorted_indices[bi, 1:][sorted_indices_to_remove[bi, :]]
        logits[bi, indices_to_remove] = -float('inf')

    # indices_to_remove = sorted_indices[..., 1:][sorted_indices_to_remove]
    # logits[..., indices_to_remove] = -float('inf')
    new_probs = torch.softmax(logits, dim=-1)

    cat = Categorical(probs=new_probs)
    index = cat.sample()
    return index


# input: logit [batch size, vocab size]
# output: indices[batch size]
def top_k_sample(logits, k=5):
    sorted_probs, sorted_indices = logits.sort(descending=True)
    to_remove = sorted_indices[:, k:]
    for bi in range(logits.size(0)):
        logits[bi, tuple(to_remove[bi])] = -float('inf')
    probs = torch.softmax(logits, dim=-1)

    cat = Categorical(probs=probs)
    index = cat.sample()
    return index


# input: numpy array [vocab size]
# output: int
def choose_from_top(probs, n=5):
    # np.argpartition: return indices, first n is smaller than others, if n is negative, last n is larger than others
    # top-k sampling
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)
