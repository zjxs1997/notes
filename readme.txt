各种杂七杂八的垃圾小代码

baidu translate是调用百度翻译的api，但是官网有链接示例代码，白写了。。。

bpe是一个很垃圾的byte pair encoding代码，不能用（笑）

pointer network目录下是一个简单的指针网络代码

torchtext:

seq2seq目录下就是处理seq2seq任务，比如机器翻译等。用了TranslationDataset这个类。自动添加sos和eos。

seq2num处理的是序列二分类的任务，比如语句情感分析等。没找到合适的类，用的也是TranslationDataset。在TARGET端用的field不进行vocab化和序列化，处理起来没有问题，就是看起来有点怪怪的。

tabular目录下也是seq2seq的任务，不过读取数据集方式不一样，用了更具有通用性的TabularDataset，这样数据的source和target不会被限制，而且也可以有多个source。

附一个报告模板report_template.tex，用xelatex编译就行了。

logger_test.py是一个简单使用logging模块的样例代码。



