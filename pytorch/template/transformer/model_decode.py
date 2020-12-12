import pickle

import en_core_web_sm
import torch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from model import build_model
from util import easy_field

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
spacy_en = en_core_web_sm.load()


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def model_decode(src, src_field, trg_field, model, device, max_len=310):
    model.eval()

    tokens = src.lower()
    tokens = tokenize_en(tokens)
    if len(tokens) > 300:
        tokens = tokens[:300]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indices = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.tensor(src_indices, dtype=torch.int64).unsqueeze(0).to(device)
    # src_tensor: [1, src len]
    src_mask = model.create_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indices = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.tensor(trg_indices, dtype=torch.int64).unsqueeze(0).to(device)
        trg_mask = model.create_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        # output: [batch size = 1, trg len, output dim]
        pred_token = output.argmax(2)[:, -1].item()
        trg_indices.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indices]

    return trg_tokens[1:], attention


def decode_path(model, tsv_path, passage_field, question_field, print_every=200, verbose=False):
    src_in = []
    trg_in = []
    for line in open(tsv_path):
        question, _, _, passage = line.strip().split("\t")
        src_in.append(passage)
        trg_in.append(question)

    trg_gold = []
    trg_pred = []
    index = 0
    for src, trg in zip(src_in, trg_in):
        trg_tokens, _ = model_decode(src, passage_field, question_field, model, device)
        trg_tokens = trg_tokens[:-1]

        trg_pred.append(trg_tokens)
        trg = tokenize_en(trg)
        trg_gold.append([trg])
        index += 1
        if verbose and index % print_every == 0:
            print(trg_tokens)
            print(trg)
            print()
    
    cb = corpus_bleu(trg_gold, trg_pred)
    if verbose:
        print(cb)
        print(trg_gold[0])
        print(trg_pred[0])
    
    return cb, trg_pred




if __name__ == "__main__":
    question_field = pickle.load(open('question_field.pt', 'rb'))
    passage_field = pickle.load(open('passage_field.pt', 'rb'))
    model = build_model(
        len(passage_field.vocab), len(question_field.vocab), device, 
        passage_field.vocab.stoi[passage_field.pad_token], question_field.vocab.stoi[question_field.pad_token]
    )
    model.load_state_dict(torch.load("best_model.pt"))

    decode_path(model, '../data/test.tsv', passage_field, question_field)

