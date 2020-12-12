# 改完了orz
# load dataset from xx_align.pkl

import pickle

import torch
from torchtext.data import BucketIterator, Dataset, Example, Field

from hparam import get_parser

def split_words(string):
    return string.split()

def load_examples_from_pickle(name, source_field, target_field, hparam,):
    # 还是弄短一点比较好，避免因为bos eos之类的出问题
    target_max_len = hparam.target_max_len-5
    alignments = pickle.load(open(f'{hparam.dataset_prefix}/{name}_align.pkl', 'rb'))

    examples = []

    for src, trg in alignments:
        src = src.lower()
        trg = trg.lower()

        source_tokens = source_field.tokenize(src)
        target_tokens = target_field.tokenize(trg)
        if len(source_tokens) > hparam.source_max_len or len(target_tokens) > target_max_len:
            continue

        example = Example()
        setattr(example, 'src', src)
        setattr(example, 'trg', trg)
        examples.append(example)

    print("剩下的examples数量：", len(examples))
    return examples


# 改完了orz
def build_dataset(hparam):
    # 这里用不用额外设置tokenize都一样，因为默认就是调用split的
    source_field = Field(tokenize=split_words, batch_first=True,)
    target_field = Field(tokenize=split_words, init_token='<bos>', eos_token='<eos>', batch_first=True, include_lengths=True)

    train_examples = load_examples_from_pickle('delete_train', source_field, target_field, hparam)
    val_examples = load_examples_from_pickle('delete_val', source_field, target_field, hparam)

    fields = [('src', source_field), ('trg', target_field)]
    train_dataset = Dataset(train_examples, fields)
    val_dataset = Dataset(val_examples, fields)

    # import ipdb; ipdb.set_trace()
    source_field.build_vocab(train_dataset, min_freq=hparam.target_min_freq)
    target_field.build_vocab(train_dataset, min_freq=hparam.target_min_freq)

    if hparam.device is not None:
        device = torch.device(hparam.device)
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    train_iterator = BucketIterator(train_dataset, batch_size=hparam.batch_size, device=device)
    valid_iterator = BucketIterator(val_dataset, batch_size=hparam.batch_size, device=device, train=False, sort=False)

    return source_field, target_field, train_iterator, valid_iterator


try:
    if __name__ == "__main__":
        arg_parser = get_parser()
        hparam = arg_parser.parse_args()
        built_data = build_dataset(hparam)
        source_field, target_field, train_iterator, valid_iterator = built_data

        print("source词表大小", len(source_field.vocab.itos))
        print("target的词表大小", len(target_field.vocab.itos))

        for bi, batch in enumerate(train_iterator):
            pass
        print('source的shape', batch.src.shape)
        trg, trg_lens = batch.trg
        print('target的shape', trg.shape)
        print('target的lens', trg_lens)
        print('target的一个样例example,转换回token:', end=' ')
        for index in trg[0].cpu().tolist():
            print(target_field.vocab.itos[index], end=' ')
        print()
        
        for bi, batch in enumerate(valid_iterator):
            pass

except Exception as e:
    __import__('ipdb').post_mortem()
