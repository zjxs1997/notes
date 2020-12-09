### 杂项笔记

#### 总体把握

Field是定义用来做各种杂七杂八的事情的，不太能用三言两语总结出来。

Example是数据最小的一个单位。

Example打包起来后就可以定义一个Dataset。

Dataset不可能直接使用，会用一个Iterator包装一下。它是一个generator，会遍历Dataset，做各种杂事，顺便把Example中的数据转换成torch的tensor格式（转化规则看Field）。

#### Field

总结一下Field的用法，我觉得主要要注意的就是use_vocab和sequential。如果单纯是数字，那use_vocab就是False。但是数字的序列的情况下需要注意，pad_token这个必须要手动设置成pad_token_id，不然最后的结果中会用`<pad>`这个string来pad，就无法转成tensor了。

Field里面的vocab，有两个字段，stoi和itos。stoi本质是个defaultdict，而itos是个list。非常需要注意的是！！！这两个东西的len返回的值不一样。因为Field在build_vocab的时候会传入min_freq参数。出现频率比这个低的都不会在itos中出现，但是，在stoi中会有，它们还会被default映射到unk_token_id。因此，初始化embedding之类的，一定要用len(itos)做参数。

Field的build_vocab函数，接受的参数是*args与**kwargs，因此传参的时候，可以传入多个Dataset实例，也就是多个数据集公用一份词表。

#### Iterator

上面也提到，因为Iterator是个generator，所以不能用pickle存到本地。因此如果数据预处理和载入用时比较长，想保存到本地以缩短时间的话，只能把Dataset和Field存起来，Iterator还得载入这两者之后手动创建。

