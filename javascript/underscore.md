## underscore
underscore和jQuery一样，也是一个第三方库，它用下划线_存各种方法。它主要提供的是各种函数式风格的方法。

- map和filter，这两个也可以作用到object上，但是返回的都是array，如果要返回object，可以用mapObject。

- every和some，顾名思义。

- max和min，，，

- groupby，把传入的元素按key归类，key由传入的函数调用得到，返回一个object，key就是函数返回的各个key，value是相同key的元素的array。

- shuffle和sample，就是python里的random.shuffle，random.choices

### 接下来这些是专用在Array上的
- first和last，，，

- flatten，可以递归把嵌套的array拍平，这个自己写还是略麻烦。

- zip和unzip，zip用过python很熟悉，unzip就是逆向的操作。

- object，有点类似zip，不过直接作为key、value拼成一个object返回了。

- range，就是python那个，不过返回的是个array。

### 用在object上的
- keys返回所有key，allKeys返回（包括原型链上继承的）所有key。

- values返回所有value。

- invert，键值翻转。

- extend，把多个object的键值对合并到第一个object，并且返回第一个。extendOwn：获取属性时忽略从原型链继承下来的属性。


### 感受
总感觉这部分有点trivial，没什么意思。。。。




