import argparse
from collections import defaultdict

def get_parser():
    arg_parser = argparse.ArgumentParser()

    # 数据处理
    arg_parser.add_argument('--source_max_len', type=int, default=300)
    arg_parser.add_argument('--target_max_len', type=int, default=120)
    arg_parser.add_argument('--target_min_freq', type=int, default=3)

    # encoder
    arg_parser.add_argument('--encode_dropout', type=float, default=0.2)
    arg_parser.add_argument('--enc_layers', type=int, default=2)
    arg_parser.add_argument('--enc_heads', type=int, default=2)
    arg_parser.add_argument('--enc_pf_dim', type=int, default=256)
    arg_parser.add_argument('--hid_dim', type=int, default=384)

    # decoder
    arg_parser.add_argument('--dec_layers', type=int, default=2)
    arg_parser.add_argument('--dec_heads', type=int, default=2)
    arg_parser.add_argument('--dec_pf_dim', type=int, default=256)
    arg_parser.add_argument('--decode_dropout', type=float, default=0.3)

    # 训练
    arg_parser.add_argument('--train_epoch', type=int, default=400)
    arg_parser.add_argument('--device', type=str, choices=['cpu', 'cuda'])
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--model_learning_rate', type=float, default=4e-4)
    arg_parser.add_argument('--gradient_accumulate', type=int, default=2)
    arg_parser.add_argument('--optim_factor', type=float, default=4.0)
    arg_parser.add_argument('--warmup_steps', type=int, default=10000)
    arg_parser.add_argument('--clip', type=float, default=1.0)

    # 保存载入与log
    arg_parser.add_argument("--load_model", type=str)
    arg_parser.add_argument("--save_path", type=str, default='result')
    arg_parser.add_argument('--print_loss_every_batch', type=int, default=20)
    arg_parser.add_argument('--dataset_prefix', type=str, default="data")


    return arg_parser

if __name__ == "__main__":
    arg_parser = get_parser()
    hparam = arg_parser.parse_args()
    print(hparam)