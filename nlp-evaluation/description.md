## BLEU

BLEU 本来是用来评价机器翻译的，而在那种情况标准的翻译（ref）可能有好几个，但是这里只讨论单个 ref 的情况。

对于 pred 和 ref，计算他们n-gram的重合度：p_n = coverage / pred

为了惩罚短序列，引入了 brief penalty：BP = 1, if pred > ref; exp(1 - pred / ref), else

最终的 BLEU 定义成：
BP * exp(w_1 * log p_1 + ...)

有时候 p_i 可能会是0，为了应对这种情况，引入了 smooth function。
具体有很多种，不过基本上都是用某种规则改变 p_n 的分子和分母。详细不展开了。


---


BLEU-n 一般是指1，2，...n-gram的 BLEU score 的几何平均。

corpus-BLEU 不等于 sentence-BLEU 的平均数，因为前者是把所有的分子分母分别相加在相除得到的。




## ROUGE

和bleu不一样，除了计算 overlap/pred（precision）之外，还要计算 overlap/ref（recall）。现在基本上都是看f-1 score的了。

一般基于n-gram的只会看到bigram。

ROUGE-L是基于最长公共子序列的。 overlap用LCS的长度替代。

ROUGE-S是基于skip-gram。而ROUGE-SU除了skip-gram之外也考虑unigram。


## METEOR

主要是用wordnet扩充了一下同义词集合，同时也考虑了英语等语言中的单词形变的问题。比如go和goes这种。不过meteor似乎只评价unigram。

不过缺点也比较明显，首先是有比较多的超参数，这个只是在某个数据集上通过grid search得到的，但可能并不适用于别的场景。另外就是比较依赖wordnet这种外部知识。

## CIDER

似乎是用onehot向量来表示两个文档，然后通过向量相似度来衡量两个文档的相关性。因为CIDER主要是用来评价图像caption之类的任务的。在这些任务中文本只需要提取出最关键的信息，即便多了或漏了一些并不关键的信息，对结果的影响也不是很大，所以这么设计的。







