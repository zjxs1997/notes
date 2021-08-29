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







