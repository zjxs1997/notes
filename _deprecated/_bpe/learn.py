import collections
import pickle
import re

import en_core_web_sm
import tqdm

spacy_en = en_core_web_sm.load()
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]



paths = [
    '/Users/xusheng/Desktop/question-generation/data/dev.passage',
]

origin_vocab = collections.defaultdict(int)
for path in paths:
    for line in open(path):
        words = tokenize_en(line.strip())
        for word in words:
            origin_vocab[word] += 1

vocab = {' '.join(w) + ' </w>':i for w, i in origin_vocab.items()}

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def subword_vocab(vocab):
    sb_vocab = collections.defaultdict(int)
    for word, i in vocab.items():
        symbols = word.split()
        for s in symbols:
            sb_vocab[s] += i
    return sb_vocab


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


sb_vocab = subword_vocab(vocab)
ori_subword_num = len(sb_vocab)
target_subword_num = 6000


num_merges = target_subword_num - ori_subword_num
# 每次merge必定导致subword词表增大1
for i in tqdm.trange(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    # print(best)

sb_vocab = subword_vocab(vocab)
pickle.dump(sb_vocab, open("subword_vocab.pkl", 'wb'))

