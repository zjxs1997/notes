import en_core_web_sm
import spacy
import torch
from torchtext.data import BucketIterator, Field, Iterator, TabularDataset
from torchtext.datasets import TranslationDataset

spacy_en = en_core_web_sm.load()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cpu_device = torch.device("cpu")


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def build_dataset(data_path='../data'):

    QUESTION = Field(
        # tokenize=lambda x: x.split(),
        tokenize=tokenize_en,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True,
        batch_first=True
    )

    TITLE = Field(
        # tokenize=lambda x: x.split(),
        tokenize=tokenize_en,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True,
        batch_first=True
    )

    PASSAGE = Field(
        # tokenize=lambda x: x.split(),
        tokenize=tokenize_en,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True,
        batch_first=True
    )

    BOOL = Field(sequential=False, use_vocab=False)


    train_dataset, val_dataset, test_dataset = TabularDataset.splits(
        path=data_path,
        fields=[('question', QUESTION), ('title', TITLE), ('answer', BOOL), ('passage', PASSAGE)],
        train='train.tsv', validation='val.tsv', test='test.tsv', format='tsv',
        filter_pred=lambda x: len(x.passage) <= 300,
    )

    QUESTION.build_vocab(train_dataset, min_freq=2)
    TITLE.build_vocab(train_dataset, min_freq=2)
    PASSAGE.build_vocab(train_dataset, min_freq=2)

    train_iterator, val_iterator, test_iterator = BucketIterator.splits(
        (train_dataset, val_dataset, test_dataset),
        batch_size = 96,
        device = cpu_device,
        sort_within_batch = True,
        sort_key = lambda x: len(x.passage),
        shuffle=True
    )

    # 后续的考虑：test iterator不用bucket可以吗，因为bucket会打乱
    return train_iterator, val_iterator, test_iterator, QUESTION, TITLE, PASSAGE, BOOL
