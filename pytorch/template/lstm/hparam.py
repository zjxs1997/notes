import argparse

def get_parser():
    arg_parser = argparse.ArgumentParser()

    # 数据预处理
    arg_parser.add_argument("--raw_data_path", type=str, default='bert_data')
    arg_parser.add_argument("--src_min_freq", type=int, default=3)
    arg_parser.add_argument("--trg_min_freq", type=int, default=3)

    # arg_parser.add_argument("--src_sep_token", type=str, default='<sep>')
    # arg_parser.add_argument("--trg_sep_token", type=str, default='<sep>')
    arg_parser.add_argument("--src_max_len", type=int, default=200)
    arg_parser.add_argument("--trg_max_len", type=int, default=100)

    arg_parser.add_argument("--batch_size", type=int, default=16)


    # 模型
    arg_parser.add_argument("--enc_emb_dim", type=int, default=300)
    arg_parser.add_argument("--enc_hid_dim", type=int, default=300)
    arg_parser.add_argument("--enc_dropout", type=float, default=0.2)
    arg_parser.add_argument("--enc_num_layer", type=int, default=2)

    arg_parser.add_argument("--dec_emb_dim", type=int, default=300)
    arg_parser.add_argument("--dec_hid_dim", type=int, default=300)
    arg_parser.add_argument("--dec_dropout", type=float, default=0.2)


    # 训练
    arg_parser.add_argument("--device", type=str, choices=['cpu', 'cuda'])
    arg_parser.add_argument("--train_epoch", type=int, default=100)
    arg_parser.add_argument("--save_path", type=str, default='bert')

    arg_parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    arg_parser.add_argument('--train_print_loss_every', type=int, default=100)

    # arg_parser.add_argument("--gradient_accumulate", type=int, default=2)

    return arg_parser
