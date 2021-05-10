# 记录一下各种预训练模型

## bert

Bert印象比较深刻，预训练任务有两个，MLM和NSP。模型的构成基本就是几个transformer encoder叠起来。

- MLM： mask language model，根据某个比例把输入中的token mask掉，或者是改成别的一个token，再或者是不变。输入通过encoder得到表征的vector，然后通过分类器预测mask的地方原来是什么token。

- NSP： next sentence prediction，把两个句子连起来，通过encoder编码后，cls token通过一个分类器，来预测这两个句子有没有前后顺序的关系。

但是据说这个任务反而是会降低模型的表现，因此在Roberta里面把这个任务删掉了。

## Pegasus

Pegasus是一个专门针对摘要做的预训练模型。有一个完整的transformer encoder & decoder结构。预训练任务是MLM和GSG。以及，在large模型中取消了MLM任务。

- GSG： Gap Sentence Generation，输入文档中，按某个比例把其中的句子mask掉，然后模型通过decoder生成这些句子。
具体这些句子怎么选择，也有讲究。通过rouge指标挑选出文档中最重要的几个句子。具体的重要法也有好几种评价方式，不展开了。


