import pickle

import torch

from hparam import get_parser
from model import build_model

parser = get_parser()
hparam = parser.parse_args()

if hparam.device is not None:
    device = torch.device(hparam.device)
else:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# 在外面model.eval与torch.no_grad
# 单纯的给定source，生成target
def decode_string(model, source_indices, src_field, trg_field, hparam,):
    src_vocab = src_field.vocab
    trg_vocab = trg_field.vocab
    trg_eos_idx = trg_vocab.stoi[trg_field.eos_token]
    trg_unk_idx = trg_vocab.stoi[trg_field.unk_token]

    src_tensor = torch.tensor(source_indices, dtype=torch.int64, device=device).unsqueeze(1)
    src_len = torch.tensor([len(source_indices)], dtype=torch.int64, device=device)

    encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len,)
    mask = model.create_mask(src_tensor)

    trg_indices = [trg_vocab.stoi[trg_field.init_token]]
    trg_tokens = [trg_field.init_token]

    trigram_set = set()

    for ti in range(hparam.trg_max_len):
        trg_tensor = torch.tensor([trg_indices[-1]]).to(device)
        model_prediction, hidden, cell, attention = model.decoder(trg_tensor, hidden, cell, encoder_outputs, mask,)

        eos_flag = False
        pred_index = model_prediction[0].argmax().item()
        for pred_index in model_prediction[0].topk(100)[1].tolist():
            if pred_index == trg_eos_idx:
                eos_flag = True
                break
            if pred_index == trg_unk_idx:
                # copy
                for src_index in attention[0].sort(descending=True)[1].cpu().tolist():
                    src_vocab_index = source_indices[src_index]
                    src_token = src_vocab.itos[src_vocab_index]
                    trg_vocab_index = trg_vocab.stoi[src_token]
                    if len(trg_tokens) >= 2:
                        new_trigram = '\t'.join(trg_tokens[-2:] + [src_token])
                        if new_trigram in trigram_set:
                            continue
                        else:
                            trigram_set.add(new_trigram)
                    trg_indices.append(trg_vocab_index)
                    trg_tokens.append(src_token)
                    break
                break
            else:
                pred_token = trg_vocab.itos[pred_index]
                if len(trg_tokens) >= 2:
                    new_trigram = '\t'.join(trg_tokens[:-2] + [pred_token])
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

    return trg_tokens, trg_indices


def decode_alignments(model, alignments, src_field, trg_field, hparam):
    src_vocab = src_field.vocab
    model.eval()
    results = []
    with torch.no_grad():
        for idx, (source, target) in enumerate(alignments):
            source = source.lower()
            target = target.lower()

            source_tokens = [src_field.tokenize(source)]
            source_indices = [src_vocab.stoi[t] for t in source_tokens]
            if len(source_indices) > hparam.src_max_len:
                source_indices = source_indices[:hparam.src_max_len]
            
            pred_trg_tokens, pred_trg_indices = decode_string(model, source_indices, src_field, trg_field, hparam,)
            pred_target = ' '.join(pred_trg_tokens)

            results.append((target, pred_target))

            if idx % 100 == 0:
                print(target)
                print(pred_target)
                print("=" * 50)
    return results


if __name__ == "__main__":

    src_field = pickle.load(open(f'{hparam.save_path}/source_field.pt', 'rb'))
    trg_field = pickle.load(open(f'{hparam.save_path}/target_field.pt', 'rb'))

    model = build_model(hparam, src_field, trg_field).to(device)
    model.load_state_dict(torch.load(f'{hparam.save_path}/checkpoint99.pt'))

    test_alignments = pickle.load(open('bert_data/delete_test_cluster_align.pkl', 'rb'))
    decode_result = decode_alignments(model, test_alignments, src_field, trg_field, hparam)

    pickle.dump(decode_result, open('bert/decode_result.pkl', 'wb'))

