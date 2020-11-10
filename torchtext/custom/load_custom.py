# 这个任务的输入和输出都是string的list，从pickle文件中读取数据。
# 默认两端的batch size都是1了，也正是利用了这一点才能魔改继承原来的field并override加了两行代码。
# Iterator中返回的对象是个二维tensor，形状为
# (src_list_len, src_seq_len) 和 target端的，因为是tensor所以肯定有padding，可以考虑写个解padding的小函数


import pickle

import torch
from torchtext.data import Field, BucketIterator, Dataset, Example

class MyField(Field):
    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.

        Pad, numericalize, and postprocess a batch and create a tensor.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
            and custom postprocessing Pipeline.
        """
        # add 2 lines
        assert len(batch) == 1
        batch = batch[0]

        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor


source_field = MyField(tokenize=lambda x: x.split(), lower=True, batch_first=True)
target_field = MyField(tokenize=lambda x: x.split(), init_token='<bos>', eos_token='<eos>', lower=True, batch_first=True)


source_list = pickle.load(open("data/source.pkl", 'rb'))
target_list = pickle.load(open("data/target.pkl", 'rb'))

examples = []
for src, trg in zip(source_list, target_list):
    example = Example()
    processed_sources = [source_field.preprocess(s) for s in src]
    processed_targets = [target_field.preprocess(t) for t in trg]
    setattr(example, 'src', processed_sources)
    setattr(example, 'trg', processed_targets)
    examples.append(example)

fields = [('src', source_field), ('trg', target_field)]
train_dataset = Dataset(examples, fields)

source_field.build_vocab(train_dataset, min_freq=1)
target_field.build_vocab(train_dataset, min_freq=1)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_iterator = BucketIterator(train_dataset, batch_size=1, device=device)
# 怀疑要自己实现batch_size_fn

for bi, batch in enumerate(train_iterator):
    src = batch.src
    trg = batch.trg

    for l in src.tolist():
        for i in l:
            print(source_field.vocab.itos[i], end=' ')
        print()

    for l in trg.tolist():
        for i in l:
            print(target_field.vocab.itos[i], end=' ')
        print()

# 封装成函数return的话得return两个field（可能只要source的）和iterator

