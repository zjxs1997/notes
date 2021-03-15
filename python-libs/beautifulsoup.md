## Beautiful Soup

这是一个可以用来解析html、以及xml文档的库，可以很好地用在爬虫脚本上。

import:`from bs4 import BeautifulSoup`；
使用：`soup = BeautifulSoup(html_text)`。

得到bs对象之后，就可以进行各种操作了。

我最喜欢的就是通过各种方式查找dom节点：

- 通过class属性值：`soup.find_all(class_='badge')`

- 待补充orz

查找之后返回的是一个tag的list，tag可以通过a.title的方式或者a['title']的方式访问属性值。不过这里要注意的是，返回的可能是它内置的一个字符串格式，使用的时候可能会出点问题，为了保险起见还是用str函数转换比较好。
