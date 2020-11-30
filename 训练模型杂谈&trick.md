### 杂谈与trick

- train模型的时候，整个epoch的循环可以用一个catch包起来，except一个KeyboardInterrupt，这样看着觉得差不多的时候可以直接ctrl+c跳出循环，执行后续的代码。
- train模型的代码可以通过命令行读取参数，得到一个suffix，保存模型与checkpoint的时候加上suffix，这样同一份代码就可以train多个模型。
```python
import sys
argv = sys
suffix = sys[1] if len(sys) >= 2 else 'suffix'
# sys[0]是执行的python文件（在命令行里敲的）名。
```
- 用torchtext库load_dataset的时候可以保存Dataset object，下次执行的时候检查是否已经有保存的了。如果有就直接load，省去了build from scratch的时间。

- pytorch的torch.utils.checkpoint还挺好用的，可以节省显存，用来开大batch size还比较实用，不过速度会稍微慢点。

- 模型参数的初始化，可以使用apply函数，逐层递归init。
- cross entropy loss可以指定ignore_index为PADDING_INDEX，这样在计算loss的时候就不会把padding加入。
- 某tutorial推荐decode的时候用len=1的LSTM，而不是自己用LSTMCell，我也不知道有什么讲究。不过LSTMCell好像确实不能指定num_layers，毕竟就是单个cell。（然而decode的LSTM似乎也不需要多层？）
- 训练模型的时候，不要只打印自动评价指标，最好随机挑一个example打印hypothesis和reference，这样才能找到问题所在。
