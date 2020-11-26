# 详见：https://github.com/pltrdy/rouge

from rouge import Rouge

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
    list_of_refs.append(" ".join(r))
    list_of_hyps.append(" ".join(h))

rouge = Rouge()
scores = rouge.get_scores(list_of_hyps, list_of_refs, avg=True)
print(scores)

