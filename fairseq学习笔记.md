## fairseq

久闻fairseq大名，现在来好好学一学。。。

就我现在看到的，fairseq的逻辑似乎是：大的框架我们给你搭好了，你要用的话，就具体实现你想实现的东西好了。这么说似乎又太模糊了。fairseq里可以手工添加的似乎就只有五类模块，其中最重要的就是model和task。

根据官网的描述，给定model，criterion，task，optimizer和lr_scheduler，fairseq的训练流程是：

```python
for epoch in range(num_epochs):
    itr = task.get_batch_iterator(task.dataset('train'))
    for num_updates, batch in enumerate(itr):
        task.train_step(batch, model, criterion, optimizer)
        average_and_clip_gradients()
        optimizer.step()
        lr_scheduler.step_update(num_updates)
    lr_scheduler.step(epoch)
```

task的train_step大致是：

```python
def train_step(self, batch, model, criterion, optimizer, **unused):
    loss = criterion(model, batch)
    optimizer.backward(loss)
    return loss
```

### 安装

fairseq安装的最好方式是下载GitHub上的repo，然后到本地`pip install` ，如果直接`pip install fairseq`似乎是会报错的。这样安装之后这个库的源文件都还在你下载解压的地方，不会到`/path/to/python/lib/python3.x-site-packges`下面。后续要改源代码/加文件都可以在那个目录下进行。

macOS上安装的时候，遇到了pyyaml的版本不对的问题，但是pyyaml似乎是内置的库不能升级。解决方法：pip安装的时候加一个`--ignore-installed PyYAML`。

### model

model的概念很容易理解，具体来说就是torch的module。model里面分好多种，都得继承`FairseqBaseModel`类。如果是一个序列分类模型，那直接继承、实现就行了。那如果是seq2seq呢，你可以先分别实现`FairseqEncoder`和`FairseqDecoder`，然后再把两者拼装起来，实现一个`FairseqEncoderDecoderModel`。如果嫌decoder跑的太慢，可以用`FairseqIncrementalDecoder`，并且实现它需要你实现的一些方法。
虽然说你得遵守它的一些规则，但其实还是有比较大的自由度的。里面会通过调用`build_model()`方法来构造模型，可以在这里做各种想做的事。model还可以override一个`add_args`函数，用来添加命令行工具可以parse的参数。
模型的源文件要放在fairseq/models下面。

写完model之后，一定要通过调用一个函数把这个model注册起来。还有，需要定义一个architecture（是个函数，接受一个args参数，可以对args里面的变量进行in-place的赋值），把arch与model绑定起来（我也不知道为什么要另外设计一个arch）。注册完之后，就可以通过命令行工具使用了。



### task

那么model具体又是怎么被调用的呢？答案是通过task。task里的东西应该比较杂，至少包含了dictionary和dataset，似乎还指定了model的输入参数等。这个官方的文档里好像也没有详细说明，我也没仔细看。官方给的例子里是直接继承了它内置的一个task，其中的dictionary啊，dataset啊，也是直接继承的。

同样的，task也可以override一个`add_args`函数。同样，源文件要放在fairseq/tasks目录下。



### 命令行工具

fairseq提供了很多命令行工具，但是感觉会让人疑惑。

#### fairseq-preprocess

显然是用来做预处理的。这个命令绑定到`path/to/python/bin/fairseq-preprocess`这个python脚本文件。兜兜转转最后跑到`path/to/fairseq/fairseq-cli/preprocess.py`这个文件下面执行它的`cli_main()`函数。

#### fairseq-train

显然是用来训练的。这个命令绑定到`path/to/python/bin/fairseq-train`这个python脚本文件。具体也不用多展开，只要知道用这个命令`fairseq-train args`和`python path/to/fairseq/fairseq-cli/train.py args`执行这个python文件效果是一样的。如果有兴趣的话可以更详细地看这个python文件内部的代码之类的。

#### fairseq-generate

用来运行一个训练好的seq2seq模型进行生成。同样，绑定到了`path/to/python/bin/fairseq-generate`这个脚本，最后则是`path/to/fairseq/fairseq-cli/generate.py`这个文件的`cli_main()`函数。

#### fairseq-eval-lm

看名字也能知道要做什么了吧。

#### 其他

略。













