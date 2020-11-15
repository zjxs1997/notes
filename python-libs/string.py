# python有个string库

import string

# 返回一个string，好像是所有英文标点，偶尔可以用用吧
print(string.punctuation)

text = 'abc., sadfsad;{afd'
# 如果想删掉一个文本text里的所有标点
# 下面这种做法貌似有点笨
new_text = ''
for t in text:
    if t not in string.punctuation:
        new_text += t
# 可以配合re使用
import re
new_text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

# 我测试的时候，发现第二种方法好像更慢一点？



# 还有个html库，爬取网页比较实用
import html

# 爬网页的时候经常会看到html源代码里有&amp &quot这种东西，可以用下面这个函数处理
a = '02&amp;d=&quot;4'
print(html.unescape(a))
