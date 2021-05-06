# 各种不方便归类的乱七八糟的操作

## 固定模块中某一子模块的参数

只需要设定该模块的requires_grad为False即可，或者直接调用方法`m.requires_grad_(False)`

## 怎么看模块参数的梯度呢？
```python
p = model.encoder.fc_out.parameters()
for param in p:
    print(p.grad)
```

## 增加embedding的维度
时不时会遇到这种情况：先在某个数据集上预训练，然后到另一个数据集上训练。如果这两个数据集的词表不一样的话，就需要改变embedding的维度。
之前看到transformers的bert里面有个`resize_token_embeddings`方法，就是用来做这个事的。看了一下它的具体实现，其实很土。。。就是新建了一个embedding层，然后把旧的参数复制过去。
```python
def doit(old_embeddings, new_num_tokens):
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device)

    _init_weights(new_embeddings)
    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
    new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
```
