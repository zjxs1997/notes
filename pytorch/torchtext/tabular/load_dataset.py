import torch
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.datasets import TranslationDataset

SRC = Field(tokenize=lambda x: x.split(), init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=lambda x: x.split(), init_token='<sos>', eos_token='<eos>', lower=True)

# train_dataset, val_dataset, test_dataset = TranslationDataset.splits(path='data', fields=(SRC, TRG), exts=('.src', '.trg'))
train_dataset, val_dataset, test_dataset = TabularDataset.splits(path='data', fields=[('chs', SRC), ('eng', TRG)], train='train.csv', validation='val.csv', test='test.csv', format='tsv')

# print(len(train_dataset.examples), len(val_dataset.examples), len(test_dataset.examples))
# 800, 100, 100

# print(vars(train_dataset.examples[0]))
# {'src': ['心', '如', '在', '，', '梦', '就', '在', '！'], 'trg': ['where', 'there', 'is', 'a', 'heart', ',', 'there', 'is', 'a', 'dream', '!']}

SRC.build_vocab(train_dataset, min_freq=2)
TRG.build_vocab(train_dataset, min_freq=2)

# print(SRC.pad_token, SRC.vocab.stoi[SRC.pad_token])
# print(TRG.pad_token, TRG.vocab.stoi[TRG.pad_token])
# <pad> 1
# <pad> 1

# print(len(SRC.vocab), len(TRG.vocab))
# 1511, 1258

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, val_dataset, test_dataset), 
    batch_size = 2, 
    device = device
)

for i, batch in enumerate(train_iterator):
    src = batch.chs
    trg = batch.eng

    # src: [src_len, batch_size] int64 tensor
    # trg: [trg_len, batch_size] int64 tensor


src = src.permute(1, 0).tolist()
for i in src[0]:
    print(SRC.vocab.itos[i], end=' ')
# <sos> 他 是 散步 还是 <unk> ？ <unk> <unk> 不 太 可能 <eos> <pad> <pad> <pad>

# for i in src[1]:
#     print(SRC.vocab.itos[i], end=' ')
# <sos> <unk> 认为 ， 一个 国家 没有 任何 理由 <unk> 或 <unk> 另一个 国家 。 <eos>


# trg = trg.permute(1, 0).tolist()
# for i in trg[0]:
#     print(TRG.vocab.itos[i], end=' ')
# <sos> did he walk or <unk> ? the <unk> <unk> <unk> <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> 
