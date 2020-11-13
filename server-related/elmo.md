#### ELMO词向量

也不知道该放哪。。。不过毕竟这个环境只在集群上有，就放这把。。。

```python
from elmoformanylangs import Embedder 
e = Embedder('/S1/LCWM/xs/models/elmo-eng/') 
sents = [
    ['Today', 'is', 'Friday', '.'], 
    ['Tomorrow', 'is', 'Saturday', ',', 'I', 'am', 'happy', '.']
] 
e.sents2elmo(sents)
```

会输出一个list，list的长度是句子数量，每个元素是numpy的array，形状是seq_len * 1024

基本不用了吧，numpy转来转去实在是麻烦。。。


