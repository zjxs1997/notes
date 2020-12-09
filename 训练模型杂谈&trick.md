### 杂谈与trick

- 训练之前可以print hparam，以确认配置写的有没有问题。

- train模型的时候，整个epoch的循环可以用一个catch包起来，except一个KeyboardInterrupt，这样看着觉得差不多的时候可以直接ctrl+c跳出循环，执行后续的代码。

- train模型的代码可以通过命令行读取参数，得到一个suffix，保存模型与checkpoint的时候加上suffix，这样同一份代码就可以train多个模型。

```python
import sys
argv = sys
suffix = sys[1] if len(sys) >= 2 else 'suffix'
# sys[0]是执行的python文件（在命令行里敲的）名。

# 不过我现在现在基本就是用argparse这个库，也可以传入suffix参数，
# 不会再用这种比较原始的方式了
```

- 保存模型checkpoint的时候也可以把optimizer同时存一下，有的optimizer可能会有warmup之类的参数，需要记录一些额外的信息。如果训练完之后，发现模型还没有完全收敛，就可以同时载入checkpoint和optimizer，快速恢复训练现场。
！！！需要注意，如果要load恢复，就一定要保证field、dataset之类的对象都保存，并且与保存的checkpoint配套。因为torchtext的build_vocab似乎不保证创建出来的词表是一样的。

- 用torchtext库load_dataset的时候可以保存Dataset object，下次执行的时候检查是否已经有保存的了。如果有就直接load，省去了build from scratch的时间。

- pytorch的torch.utils.checkpoint还挺好用的，可以节省显存，用来开大batch size还比较实用，不过速度会稍微慢点。

- 模型参数的初始化，可以使用apply函数，逐层递归init。

- cross entropy loss可以指定ignore_index为PADDING_INDEX，这样在计算loss的时候就不会把padding加入。

- 某tutorial推荐decode的时候用len=1的LSTM，而不是自己用LSTMCell，我也不知道有什么讲究。不过LSTMCell好像确实不能指定num_layers，毕竟就是单个cell。（然而decode的LSTM似乎也不需要多层？）

- 训练模型的时候，不要只打印自动评价指标，最好随机挑一个example打印hypothesis和reference，这样才能找到问题所在。

- clip_grad_norm_还是一个比较有用的点，虽然可能很烦，但是最好还是写一下。

- 训练过程可以用一个try except KeyboardInterruption包起来，这样的好处是，看log的时候人工确认到训练差不多了、收敛了的时候，可以ctrl+c直接退出，然后执行后续的，比如evaluate的操作。

#### transformer的训练心得（？）

- batch size一定要开大，如果显存不够大，也可以退而求其次，用gradient accumulate

- 要用warmup，现在有一个NoamOpt的代码，是在原始的torch optimizer的基础上的包装代码。


#### 代码模式

因为Field调用build_vocab得到的词表似乎会有一定的随机性，所以如果要继续训练保存到磁盘的模型，就一定要保证新得到的field和原来的field一样。也就是说，模型、数据、field这三者是要绑定在一起的。因此可以采用这样的模式：

- load_dataset.py文件中有两个函数build_dataset_from_scratch，载入原始数据，创建dataset对象和field对象，然后得到相应的词表；第二个函数build_iterator给定dataset创建iterator（这个其实比较简单，用几行代码就行）。load_dataset在main里面调用第一个函数，并且把field和dataset都保存到本地。

- train_model.py文件，载入field和dataset，创建模型，保存的checkpoint和field等数据都放同一个目录。


