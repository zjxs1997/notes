# pytorch module的state dict

一般为了保存模型训练结果，都是用`torch.save(model.state_dict(), 'xxx')`的方式保存state_dict。在要用的时候通过`model.load_state_dict(torch.load('xxx'))`读入并加载。
但我state_dict的机制细节等不是很了解。

## state_dict

module的state_dict函数返回的是一个OrderedDict，key是模块中的各个子模块。比如：

```python
class Model(nn.Module):
    def __init__(self):
        self.a = nn.Linear(2, 3)
        self.b = nn.Linear(3, 2)
```

那么这个类实例化后的模型，OrderedDict中的key就有`'a.weight', 'a.bias', 'b.weight', 'b.bias'`这四个。

对应的value则是这些子模块的值tensor。
这些value貌似require_grad的值都是False，然后device对应的是模型的device。

## load_state_dict

除了self之外，这个函数实际上还有两个参数，一个是state_dict，另一个是strict。
strict是个bool值，控制是否采用严格模式。如果是True，那么必须保证作为参数的state_dict的key，和模型的state_dict函数返回的OrderedDict的key完全相同。也就是说，不能有多余的key，或缺失的key。
但如果是False，则允许这种情况发生。只是最后会返回一个IncompatibleKeys对象，提示缺失了哪些key，多了哪些key。

另外就是，state_dict中tensor的device和model的device就算不一样，好像也没问题。。。

## 实际应用

现在假想这样一个场景：要使用一批不一样的数据，用一个不一样的任务预训练一个模型。和原始模型相比很多组件都相同。

在这种情况下，必须单独写一个load函数，主要是对预训练模型的state_dict做一些筛选工作，提取出需要在后续中用到的项。
另外必须注意的是，embedding的值必须要小心对待。
因为两个模型处理的数据集不一样，词表几乎可以说肯定不同。所以vocab是必须保存下来的，以便和原始模型的vocab做比较、生成新的embedding参数。
还有就是position embedding等，也要特殊处理一下，不过比token embedding要方便。
最后就是decoder最后需要通过一个Linear层，这个层的输出dim是vocab的size。

---
额，怎么感觉写了一堆废话


另外可以看看这个[链接](https://zhuanlan.zhihu.com/p/98563721)

## 顺便记录一下常见module的state_dict

- Linear: bias, weight

- Embedding: weight

- 额，别的好像都蛮复杂的，orz。算了，有必要的时候自己定义一个看看好了
