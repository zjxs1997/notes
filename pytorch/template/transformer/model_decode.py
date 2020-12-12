# 改完了orz

import pickle
import random

import torch
from hparam import get_parser

from model import build_model


def decode(model, source_field, target_field, source_indices, device, maxlen=120,):
    source_vocab = source_field.vocab
    trg_vocab = target_field.vocab
    trg_eos_idx = trg_vocab.stoi[target_field.eos_token]
    trg_unk_idx = trg_vocab.stoi[target_field.unk_token]
    src_tensor = torch.tensor(source_indices, dtype=torch.int64, device=device).unsqueeze(0)
    enc_src = model.encoder(src_tensor,)
    src_mask = model.encoder.create_src_mask(src_tensor)
    
    trg_indices = [trg_vocab.stoi[target_field.init_token]]
    trg_tokens = [target_field.init_token]

    trigram_set = set()

    for i in range(maxlen):
        trg_tensor = torch.tensor(trg_indices, dtype=torch.int64, device=device).unsqueeze(0)
        trg_mask = model.decoder.create_trg_mask(trg_tensor)
        output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        # output: [batch size=1, seq len, vocab size]
        # attention: [batch size=1, num heads, trg len, src len]

        eos_flag = False
        pred_index = output.argmax(2)[:, -1].item()
        for pred_index in output[0, -1].topk(100)[1].tolist():
            if pred_index == trg_eos_idx:
                eos_flag = True
                break
            if pred_index == trg_unk_idx:
                # copy
                head_idx = random.randrange(0, attention.size(1))
                # attention = attention[0, head_idx, -1]
                that_attention = attention[0, head_idx, -1]
                # for copy_src_index in attention.topk(100)[1].tolist():
                for src_index in that_attention.sort(descending=True)[1].cpu().tolist():
                    copy_src_index = source_indices[src_index]

                    src_token = source_vocab.itos[copy_src_index]
                    copy_trg_index = target_vocab.stoi[src_token]
                    if len(trg_tokens) >= 2:
                        new_trigram = '\t'.join(trg_tokens[-2:] + [src_token])
                        if new_trigram in trigram_set:
                            continue
                        else:
                            trigram_set.add(new_trigram)
                    trg_indices.append(copy_trg_index)
                    trg_tokens.append(src_token)
                    break
                break
            else:
                pred_token = trg_vocab.itos[pred_index]
                if len(trg_tokens) >= 2:
                    new_trigram = '\t'.join(trg_tokens[-2:] + [pred_token])
                    if new_trigram in trigram_set:
                        continue
                    else:
                        trigram_set.add(new_trigram)
                trg_indices.append(pred_index)
                trg_tokens.append(pred_token)
                break

        if eos_flag:
            break

    trg_tokens = trg_tokens[1:]
    trg_indices = trg_indices[1:]
    # 返回的tokens里不包含bos
    return trg_tokens, trg_indices


# 返回一个list,每个元素是个tuple,第一个是gold,第二个是predict
def decode_alignments(model, source_field, target_field, test_alignments, hparam, device, maxlen=120):
    source_vocab = source_field.vocab
    model.eval()
    results = []
    idx = 0
    with torch.no_grad():
        for src, trg in test_alignments:
            src = src.lower()
            trg = trg.lower()

            source_tokens = source_field.tokenize(src)
            source_indices = [source_vocab.stoi[token] for token in source_tokens]

            if len(source_indices) > hparam.source_max_len:
                source_indices = source_indices[:hparam.source_max_len-1]

            pred_tokens, pred_indices = decode(model, source_field, target_field, source_indices, device, maxlen=hparam.target_max_len)
            pred_trg = ' '.join(pred_tokens)

            results.append((trg, pred_trg))
            if idx % 100 == 0:
                print(trg, )
                print(pred_trg)
                print("-" * 50)
            idx += 1
    return results


arg_parser = get_parser()
hparam = arg_parser.parse_args()

if hparam.device is None:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device(hparam.device)


try:
    if __name__ == "__main__":
        save_path = hparam.save_path
        source_field = pickle.load(open(f"{save_path}/source_field.pkl", 'rb'))
        target_field = pickle.load(open(f"{save_path}/target_field.pkl", 'rb'))
        target_vocab = target_field.vocab

        trg_pad_idx = target_vocab.stoi[target_field.pad_token]

        model = build_model(hparam, source_field, target_field, device)
        model.load_state_dict(torch.load(f'{save_path}/checkpoint399.pt'))

        test_alignments = pickle.load(open(f'{hparam.dataset_prefix}/delete_test_align.pkl', 'rb'))
        
        results = decode_alignments(model, source_field, target_field, test_alignments, hparam, device)
        result = random.choice(results)
        print(result)

        pickle.dump(results, open(f'{save_path}/decode_result.pkl', 'wb'))


except Exception as e:
    print(e)
    __import__('ipdb').post_mortem()
