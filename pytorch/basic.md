### pytorch基本操作

- stack与cat：stack会增加一个维度，cat不会。

- 应对梯度爆炸：clip
```python
from torch.nn.utils import clip_grad_norm_
for p in model.parameters():
    print(p.grad)
    clip_grad_norm_(p, 10)
```

- 矩阵乘法matmul、mm与bmm
    - mm就是单纯的矩阵乘法，参数1是(n,m)，参数2是(m,p)维的tensor，返回(n,p)的tensor
    - bmm是带batch的矩阵乘法，参数1是(b,n,m)，参数2是(b,m,p)，返回(b,n,p)的tensor
    - matmul比较复杂，我的理解是，两个输入参数，只用最后两个维度做矩阵乘法，然后其他的维度用广播的机制得到结果（如果第一个参数维度只有1的话自动unsqueeze一个）

- torch.where：输入condition、x和y，当condition为True的时候取x的值，否则取y的值，可以用来构造一些奇妙的loss。
