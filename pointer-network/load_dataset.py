import en_core_web_sm
import torch
from torchtext.data import BucketIterator, Field, Iterator, TabularDataset
from torchtext.datasets import TranslationDataset


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cpu_device = torch.device("cpu")

spacy_en = en_core_web_sm.load()

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def build_dataset(data_path='data'):
    word_field = Field(
        tokenize=tokenize_en,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True,
        include_lengths=True,
    )
    bool_field = Field(sequential=False, use_vocab=False)

    train_dataset, val_dataset, test_dataset = TabularDataset.splits(
        path=data_path,
        fields=[('question', word_field), ('title', word_field), ('answer', bool_field), ('passage', word_field)],
        train='train.tsv', validation='val.tsv', test='test.tsv', format='tsv',
        filter_pred=lambda x: len(x.passage) <= 250,
    )

    word_field.build_vocab(train_dataset, min_freq=2)

    # todo: 测试一下，如果device选gpu会不会爆显存
    train_iterator, val_iterator, test_iterator = BucketIterator.splits(
        (train_dataset, val_dataset, test_dataset),
        batch_size = 32,
        device = cpu_device,
        sort_within_batch = True,
        # sort_key = lambda x: len(spacy_en.tokenizer(x.passage[0]))
        sort_key = lambda x: len(x.passage)
    )

    # 后续的考虑：test iterator不用bucket可以吗，因为bucket会打乱
    print(len(train_iterator.data()), len(val_iterator.data()), len(test_iterator.data()))
    return train_iterator, val_iterator, test_iterator, word_field, bool_field
