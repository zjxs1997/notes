# 新版本，val dataset中额外添加的几句话似乎会带来奇妙的表现

import pickle

import torch
from torchtext.data import BucketIterator, Dataset, Example, Field

from hparam import get_parser

def load_examples_from_pickle(name, hparam, source_field, target_field):
    # 目前这个版本没有手动添加bos与eos，等iterator跑出来了看看结果如何再作调整
    # 似乎是没有问题的
    alignments = pickle.load(open(f'{hparam.raw_data_path}/delete_{name}_cluster_align.pkl', 'rb'))
    examples = []

    for source, target in alignments:
        source = source.lower()
        target = target.lower()
        source_list = source_field.tokenize(source)
        target_list = target_field.tokenize(target)
        if len(source_list) >  hparam.src_max_len or len(target_field) > hparam.trg_max_len:
            continue

        example = Example()
        setattr(example, 'src', source_list)
        setattr(example, 'trg', target_list)
        examples.append(example)

    print(f"example num: {len(examples)}")
    return examples


def build_dataset(hparam):
    src_field = Field(lower=True, include_lengths=True)
    trg_field = Field(lower=True, init_token='<bos>', eos_token='<eos>')

    train_examples = load_examples_from_pickle('train', hparam, src_field, trg_field)
    val_examples = load_examples_from_pickle('val', hparam, src_field, trg_field)

    fields = [('src', src_field), ('trg', trg_field)]
    train_dataset = Dataset(train_examples, fields)
    val_dataset = Dataset(val_examples, fields)

    src_field.build_vocab(train_examples, min_freq=hparam.src_min_freq)
    trg_field.build_vocab(train_examples, min_freq=hparam.trg_min_freq)

    if hparam.device is not None:
        device = torch.device(hparam.device)
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    def bucket_helper(dataset):
        return BucketIterator(
            dataset, 
            batch_size=hparam.batch_size, 
            device=device,
            sort_within_batch=True,
            sort_key=lambda x:len(x.src)
        )
    
    train_iterator = bucket_helper(train_dataset)
    val_iterator = bucket_helper(val_dataset)

    return src_field, trg_field, train_iterator, val_iterator


if __name__ == "__main__":
    parser = get_parser()
    hparam = parser.parse_args()
    src_field, trg_field, train_iterator, val_iterator = build_dataset(hparam)

    for bi, batch in enumerate(train_iterator):
        break
    print(bi)
    print(batch.src, batch.trg)
    print(batch.src[0].shape)
    print("=" * 50)

    for bi, batch in enumerate(val_iterator):
        break
    print(bi)
    print(batch.src, batch.trg)
    print(batch.src[0].shape)


