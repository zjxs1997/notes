#### hugging face与各种预训练模型

hugging face有个transformers库，提供了很多预训练模型和utility。

用hugging face的模型，如果是用from_pretrain函数初始化的话，是会从网上下载到本地/tmp目录后再使用的。这样显然效率非常低，可以考虑预先下载到本地，然后从本地load。

比如说使用xlnet模型。在用from_pretrain下载的时候，一般会输出一些信息，包含了下载的url。挂个代理手动下载一下，然后放到某个目录下，比如`~/.xlnet`，之后就可以从该目录load了。注意load模型的时候，需要目录下包含两个文件，一是config.json，另一个是pytorch_model.bin。（魔改config.json可以改变模型的某些表现，比如是否输出attention等）

但如果是load tokenizer的话，只需要有个spiece.model文件即可，应该是词表文件？或者就是有一个vocab.txt文件。当然，也可以有config.json文件。

现在HuggingFace网上的文档好像很喜欢用AutoModel和AutoTokenizer这两个类，在用AutoTokenizer的时候，目录下*一定*要有config.json。

此外，transformers库里还提供了优化器AdamW，可以用来替代torch.optim.Adam？

---

以下是一些样例代码（这些代码的load路径都是集群上的）：

- xlnet
```python
import torch
from transformers import XLNetTokenizer, XLNetModel

tokenizer = XLNetTokenizer.from_pretrained('/home/xus/.xlnet/tokenizer')
model = XLNetModel.from_pretrained('/home/xus/.xlnet/base')

input_ids = torch.tensor(tokenizer.encode(
    "Hello, my dog is cute", add_special_tokens=True
)).unsqueeze(0)
outputs = model(input_ids)
last_hidden_states = outputs[0]
# The last hidden-state is the first element of the output tuple
```
这段代码最后输出的tensor是形状为(batch size, seq len, 768)的。tokenizer encode的时候指定add_special_tokens为true会在后面加入cls和sep token

- bert
```python
from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('/home/xus/.bert/tokenizer/vocab.txt')
model = BertModel.from_pretrained('/home/xus/.bert/base/')

input_ids = torch.tensor(tokenizer.encode(
    "Hello, my dog is cute", add_special_tokens=True
)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)

last_hidden_states = outputs[0]
# The last hidden-state is the first element of the output tuple
```
最后输出的tensor形状同样是(batch size, seq len, 768)。tokenizer encode的时候指定add_special_tokens为true会分别在前后加入cls和sep token

- roberta
```python
from transformers import RobertaModel, RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('/home/xus/.roberta/tokenizer')
# tokenizer可以用encode和decode方法，更快，
# 其中encode可以指定add_special_tokens参数为True

model = RobertaModel.from_pretrained('/home/xus/.roberta/base')
# RobertaModel重写（override）了BertModel类，所以用法也应该一样，
# 输入[batch size, seq len]的tensor，
# 输出是个tuple，第一个元素是[batch size, seq len, 768]，
# 第二个元素是[batch size, 768]
```
**注意：Roberta的某些token的id和Bert不一样，不能直接抄代码，而且用的是`<s> </s>`的体系，而不像bert一样是cls与sep**

- GPT2
同样也有tokenizer和model，详细使用方式可以参考[这个链接](https://huggingface.co/transformers/quickstart.html#openai-gpt-2)  （using the past这部分很有用）或者集群的~/jupyter_test/roberta_gpt2.ipynb

gpt2的最大编码长度为1024，比bert系的512大了不少。

gpt2的special token只有三个，分别是bos，eos和unk，而且这三个token还长得一模一样。没有padding token。

方法，手动添加：
```python
special_tokens_dict = {
    'bos_token': '<BOS>', 
    'eos_token': '<EOS>', 
    'pad_token': '<PAD>'
}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# return 3
# 此时len(tokenizer)从原来的50257变成了50260，
# tokenizer.eos_token等都变成了新的token
# 当然也要修改model的embedding层
model.resize_token_embeddings(len(tokenizer))
# 此时，model的get_input_embeddings与get_output_embeddings得到的
# 分别是50260的embedding层与(768, 50260)的全连接层，
# 这些新出来的参数需要finetune
```
gpt2是由transformer decoder堆叠而成的，所以应该是一个token接一个token进行生成。我现在的方式是用`src+<bos>+trg+<eos>`这个拼接后的序列作为data，（同时shift right作为目标），然后在gpt2上做语言模型的finetune。

生成的时候，用`src+<bos>`作为最初的输入，然后一个一个生成token。

理论上来说，loss只需要`<bos>`的输出以及之后的一些内容，但是如果把labels也输入进去，是全部都会计算loss的（包括bos之前的，以及pad）。看了下forward的代码，其实loss就是用交叉熵通过logit生成的，而logit是返回的，所以其实可以手动在外面操作loss。把`src+<bos>`这部分也pad即可。但是自己生成不含`src+<bos>+<pad>`的loss与直接使用返回的loss，结果貌似没什么区别。

看到一个paper是src连着trg一起训练语言模型的。



