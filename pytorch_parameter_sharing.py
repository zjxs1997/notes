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

