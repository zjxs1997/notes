import torch
import torch.nn as nn

para = nn.Parameter(torch.rand(2, 10))
input = torch.tensor([0,1,1,1,0])

emb = nn.functional.embedding(input, para)
print(emb.shape)
print(emb)

print("=" * 30)

logit = nn.functional.linear(emb, para)
print(logit.shape)
print(logit)

# =========================
# 今天突然得知还有个更简便的方式
# 引用共享/拷贝新数据之类的，感觉时不时还是不太记得清楚
emb = nn.Embedding(10, 300)
l = nn.Linear(300, 10)
l.weight = emb.weight


