# 详见：https://github.com/pltrdy/rouge
# 看了源代码，貌似是只支持rouge1、rouge2和rougeL
from rouge import Rouge

# 这个是分词，其实无所谓
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


# ---------------------------------
# ------------- 分割线 -------------
# ---------------------------------


# 详见：https://pypi.org/project/rouge-score/
# 看了代码，貌似rouge1-9都行；其他还有rougeL、以及貌似还有个基于文档级别的rougeL
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score('sdf', 'sdf dfds')
# {'rouge1': Score(precision=0.5, recall=1.0, fmeasure=0.6666666666666666),
#  'rouge2': Score(precision=0.0, recall=0.0, fmeasure=0.0),
#  'rougeL': Score(precision=0.5, recall=1.0, fmeasure=0.6666666666666666)}
# 用scores['rouge1'][2]即可提出rouge1的f1 score


# 详见：https://github.com/pltrdy/files2rouge
# 不是，这玩意我还不知道怎么用。。。
import files2rouge

files2rouge.run()


# ---------------------------------
# ------------- 分割线 -------------
# ---------------------------------




# 详见：https://github.com/li-plus/rouge-metric
# 和：https://pypi.org/project/rouge-metric/
# 可以看看源代码，里面有例子

from rouge_metric import PyRouge
rouge = PyRouge(rouge_n=(1, 2), rouge_l=True, rouge_w=False, rouge_s=False, rouge_su=True, skip_gap=4)

# 这个写法我看的有点迷惑orz
hypothesis = [['hello world'.split()]]
references = [[['hello , man'.split()]]]
scores = rouge.evaluate_tokenized(hypothesis, references)
print(scores)
# {'rouge-1': {'r': 0.3333333333333333, 'p': 0.5, 'f': 0.4}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.3333333333333333, 'p': 0.5, 'f': 0.4}, 'rouge-su4': {'r': 0.2, 'p': 0.5, 'f': 0.28571428571428575}}

# 应该是多样本的写法，refs里面每个list元素表示的是多个ref
hyps = ['hello world']
refs = [['hello , man']]
s = rouge.evaluate(hyps, refs)
print(s)
# {'rouge-1': {'r': 0.3333333333333333, 'p': 0.5, 'f': 0.4}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.3333333333333333, 'p': 0.5, 'f': 0.4}, 'rouge-su4': {'r': 0.2, 'p': 0.5, 'f': 0.28571428571428575}}

