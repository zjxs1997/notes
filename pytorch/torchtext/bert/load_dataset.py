from transformers import AutoTokenizer

from torchtext.data import Field, Example, Dataset, BucketIterator

bert_tokenizer_path = '/home/xusheng/.scibert/tokenizer'
tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_path)

train_data = [
    'yet this is a sample sentence.',
    'and here goes another example.'
]

# 一定要注意这里的pad_token
field = Field(sequential=True, use_vocab=False, batch_first=True, pad_token=tokenizer.pad_token_id)
examples = []
for data in train_data:
    example = Example()
    setattr(example, 'data', tokenizer.encode(data))
    examples.append(example)
dataset = Dataset(examples, [('data', field)])
iterator = BucketIterator(dataset, batch_size=2)
for bi, batch in enumerate(iterator):
    pass
print(bi, batch.data)
"""
输出：
0 tensor([[ 102,  137, 1530, 9289, 1783, 1143,  205,  103,    0],
        [ 102, 3481,  238,  165,  106, 1498, 8517,  205,  103]])
"""
