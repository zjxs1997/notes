import pickle
import random

import en_core_web_sm
import torch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from model import build_model
from util import easy_field


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
spacy_en = en_core_web_sm.load()

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def model_decode(src_tokens, title_tokens, src_field, title_field, trg_field, model, device, max_len=300):
    model.eval()

    # tokens = src.lower().split()
    # tokens = tokenize_en(src.lower())
    tokens = [src_field.init_token] + src_tokens + [src_field.eos_token]
    title_tokens = [title_field.init_token] + title_tokens + [title_field.eos_token]
    
    src_indices = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.tensor(src_indices, dtype=torch.int64).unsqueeze(1).to(device)
    src_len = torch.tensor([len(src_indices)], dtype=torch.int64).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)
    
    mask = model.create_mask(src_tensor)

    trg_indices = [trg_field.vocab.stoi[trg_field.init_token]]

    attentions = torch.zeros(max_len, 1, len(src_indices)).to(device)

    for ti in range(max_len):
        trg_tensor = torch.tensor([trg_indices[-1]]).to(device)

        with torch.no_grad():
            output, hidden, attention, = model.decoder(src_tensor, trg_tensor, hidden, encoder_outputs, mask)

        attentions[ti] = attention
        
        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indices]
    attentions = attentions[:len(trg_tokens) - 1]

    return trg_tokens[1:], attentions


def eval_bleu(model, src_in, title_in, trg_in, passage_field, title_field, question_field, verbose=False):
    trg_gold = []
    trg_pred = []
    index = 0
    for src, title, trg in zip(src_in, title_in, trg_in):
        src_tokens = tokenize_en(src.lower())
        title_tokens = tokenize_en(title.lower())
        trg_tokens, attention = model_decode(src_tokens, title_tokens, passage_field, title_field, question_field, model, device)
        trg_tokens = trg_tokens[:-1]

        # copy
        attention = attention.squeeze(1)
        attention_indices = attention.argmax(1)

        for ti, token in enumerate(trg_tokens):
            if token == question_field.unk_token:
                max_attn_index = attention_indices[ti].item()
                trg_tokens[ti] = src_tokens[max_attn_index-1]


        trg_pred.append(trg_tokens)
        # trg_gold.append([trg.split()])
        trg_gold.append([tokenize_en(trg.lower())])
        index += 1
        if verbose and index % 1000 == 0:
            print(trg_tokens)
            print(trg.split())
            print()
    
    cb = corpus_bleu(trg_gold, trg_pred)
    if verbose:
        print(trg_gold[0])
        print(trg_pred[0])
    return cb, trg_pred



if __name__ == "__main__":
    passage_field = pickle.load(open('word_field.pt', 'rb'))
    title_field = passage_field
    question_field = passage_field
    # model = build_model(len(ENG.vocab), len(ENG.vocab), ENG.vocab.stoi[ENG.pad_token], device)
    model = build_model(
        len(passage_field.vocab), len(question_field.vocab),
        passage_field.vocab.stoi[passage_field.pad_token], device
    )
    model.load_state_dict(torch.load("best_modell.pt"))

    src_in = []
    title_in = []
    trg_in = []
    for line in open("data/test.tsv"):
        question, title, _, passage = line.strip().split("\t")
        src_in.append(passage)
        title_in.append(title)
        trg_in.append(question)

    # src_in = open("../data/test.src").readlines()
    # trg_in = open('../data/test.trg').readlines()

    cb, trg_pred = eval_bleu(model, src_in, title_in, trg_in, passage_field, title_field, question_field, verbose=True)

    print(cb)
