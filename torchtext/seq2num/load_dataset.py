import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset

SRC = Field(tokenize=lambda x: x.split(), lower=True)
TRG = Field(sequential=False, use_vocab=False, dtype=torch.float32)

train_dataset, val_dataset, test_dataset = TranslationDataset.splits(path='data', fields=(SRC, TRG), exts=('.src', '.num'))

# print(len(train_dataset.examples), len(val_dataset.examples), len(test_dataset.examples))
# 800, 100, 100

# print(vars(train_dataset.examples[0]))
# {'src': ['心', '如', '在', '，', '梦', '就', '在', '！'], 'trg': '0.48527706951555827'}

SRC.build_vocab(train_dataset, min_freq=2)
# TRG.build_vocab(train_dataset, min_freq=2)

# print(SRC.pad_token, SRC.vocab.stoi[SRC.pad_token])
# <pad> 1

# print(len(SRC.vocab))
# 1511

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, val_dataset, test_dataset), 
    batch_size = 2, 
    device = device
)

for i, batch in enumerate(train_iterator):
    src = batch.src
    trg = batch.trg

    # src: [src_len, batch_size] int64 tensor
    # trg: [batch_size] float32 tensor


src = src.permute(1, 0).tolist()
# for i in src[0]:
#     print(SRC.vocab.itos[i], end=' ')
# 真诚 的 爱情 永远 不是 一 条 平坦 的 <unk> . <pad> <pad> <pad> 

# for i in src[1]:
#     print(SRC.vocab.itos[i], end=' ')
# 他 必须 <unk> 读书 以 积 <unk> 知识 以便 将来 服务 于 国家 。 
