# 详见：https://github.com/pltrdy/rouge
# 貌似是rouge1、rouge2和rougeL
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


# 详见：https://pypi.org/project/rouge-score/
# 看了代码，貌似rouge1-9都行；其他还有rougeL、以及貌似还有个基于文档级别的rougeL
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score('sdf', 'sdf dfds')
# {'rouge1': Score(precision=0.5, recall=1.0, fmeasure=0.6666666666666666),
#  'rouge2': Score(precision=0.0, recall=0.0, fmeasure=0.0),
#  'rougeL': Score(precision=0.5, recall=1.0, fmeasure=0.6666666666666666)}
# 用scores['rouge1'][2]即可提出rouge1的f1 score

