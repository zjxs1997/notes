# pytorch的scheduler

用来动态调整优化器的学习率，一般的使用模式都是：

```python
for ei in range(epoch):
    train()
    val()
    scheduler.step()
```

看了下几个scheduler：

- LambdaLR：接受一个函数lr_lambda，以epoch为输出，输出一个倍率，step之后优化器的学习率为原始学习率乘以该倍率。
可以给优化器中不同参数组设置不同的lr_lambda函数。

- MultiplicativeLR：与前面的不同之处在于，新的学习率是上一次的学习率乘以倍率，而不是原始的学习率乘以倍率。

上面这两种算一个系列的，因为都是给定一个函数来指定倍率，并且可以根据参数组设定不同的函数。
实际测试的时候发现，构建scheduler的时候，就会调整optimizer中的学习率。也就是会保证第0个epoch，lr全程是f(0)的值，……

- StepLR：这个相对就没那么自由了。
可以设定的参数是step_size和gamma，表示每隔多少个step（一般就是epoch），学习率就乘以gamma。

- MultiStepLR：与上一个比较类似，但不是每隔多少个step，学习率乘以gamma。
可以通过设置milestones参数设置乘以gamma的step数，比如`milestones=[30, 80]`，那就是到30个step的时候乘一次，80个step的时候再乘一次。

- ExponentialLR：我理解成就是step_size=1的StepLR

上面这三种又可以归成一类，都是每隔多少个step乘以某个固定的比例。而且这些都是对全体参数组使用的，不能分别设定。

- ReduceLROnPlateau：这个比较有意思。
它的基本思路就是，如果训练过程中，val的loss不下降了（或者别的评价指标不再往好的方向发展了），那就意味着来到了Plateau，这种时候需要降低学习率。
大概是出于可以避免过拟合等的原因。
参数列表比较复杂，有这些：

  - mode：如果mode的值是min，则表示衡量的指标是要追求越小越好，反正如果是max则追求指标更大。

  - factor：和上面那一类scheduler同样，只能传入一个定值，让新的学习率是旧的乘以该值

  - patience：可以忍受的step数。
  比如patience是2，则可以容忍前两个step的loss都不下降（默认mode是min了，下同），直到第三个step也没下降时，才采取行动降低lr。

  - threshold_mode与threshold。前者可以选rel或abs。若是rel，则新的目标值是旧的`best_loss * (1-threshold)`；否则是绝对的`best_loss - threshold`。

  -min_lr，显然是lr可以降低到的最低值。

  - eps，lr降低的最小幅度。如果新旧lr之差小于这个数值，则不更新lr。

以上也只选了一些我认为比较重要的。由于这个scheduler要比较loss指标，所以在step的时候要传入val_loss作为参数。

