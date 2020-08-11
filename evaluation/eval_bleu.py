from nltk.translate.bleu_score import corpus_bleu

import en_core_web_sm
spacy_en = en_core_web_sm.load()
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

ref_path = ""
hyp_path = ""

list_of_refs = []
list_of_hyps = []

for r, h in zip(open(ref_path), open(hyp_path)):
    r = tokenize_en(r.strip())
    h = tokenize_en(h.strip())
    list_of_refs.append([r])
    list_of_hyps.append(h)

bleu_score = corpus_bleu(list_of_refs, list_of_hyps)


