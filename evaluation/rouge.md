### 原来Rouge花样这么多

最常见的rouge-n，就是对比gold和predict的n-gram的重合度。比如有个gold`A B C D`，predict是`A b C D`，那么rouge-1的值就是重合的unigram数/gold的unigram数=3/4；rouge-2则是1/3。
如果有多个gold，则是将分子分母分别相加，然后再做除法。

rouge-L，也挺常见。是基于gold与predict的最长公共子串的。分别计算recall=最长公共子串长度/gold的长度，precision=最长公共子串长度/predict长度，F-score为两者调和平均。如果还是看上面那个例子，则最长公共子串为`A C D`长度为3，最终F-score为3/4。
文档级别的rouge-L好像有更复杂的规则，（有必要吗）

rouge-W，好像是有加权之类的，基于rouge-L的，具体还没了解清楚，暂时不写。

rouge-S是基于skip-bigram的。何谓skip-bigram？就是按照句子顺序中任何成对的词语，比如上面那个gold里就有`AB AC AD BC BD CD`这6个skip-bigram。和rouge-L一样，计算recall=重合的skip-bigram个数/gold的skip-bigram个数；precision=...，最后计算F-score。上面例子中重合的skip-bigram个数为3，rouge-S值为1/2。
因为这样的计算复杂度是n^2，开销巨大，可以限制最大跳跃距离，比如rouge-S4表示最大跳跃距离是4的rouge-S值。

rouge-SU，在rouge-S的基础上，有增加了unigram。（wiki上这么说的，具体我还不知道怎么搞。）

------

以上都是很粗略的讨论，没有涉及到重复词出现，以及重合为0的情况（这种情况下常见的做法是用一个很小的数值替代0）。

