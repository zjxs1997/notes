### 令人迷惑的python import

最近被python的import折磨的生活不能自己。。。

例如我现在有这么一个文件树：

顶层是个test文件，文件下面有个包叫mypackage，下面有a和b两个目录，a下面有spam.py和foo.py；b下面有bar.py

代码如下：
spam.py
```python
def spam():
    print('spam')
```

foo.py
```python
from .spam import spam
from ..b.bar import bar

def foo():
    spam()
    bar()
    print("foo")

if __name__ == '__main__':
    foo()
    print(1)
```

bar.py
```python
def bar():
    print("bar")
```

然后我在test顶层目录下创建一个c.py
```python
import mypackage.a.foo as foo
foo.foo()
```

接下来就是迷惑的部分了。

在test目录下执行c.py。没有任何问题。但是不论是在该目录下`python mypackage/a/foo.py`还是在a目录下执行`python foo.py`都会报错，显示的原因是

`ModuleNotFoundError: No module named '__main__.spam'; '__main__' is not a package`

也就是说，在执行foo.py的时候，相对import不work了。

那如果要让它work起来，我可以很简单的把那行`from .spam import spam`替换成`from spam import spam`，但是要对`from ..b.bar import bar`无能为力。这样可以通过`python path/to/foo.py`的方式成功调用spam函数并执行文件，但是bar是不行的。

要解决这个问题有一个方法，就是用`python -m`。还是用没有修改import的原来的代码。在test目录下执行`python -m mypackage.a.foo`这样就可以顺利执行。

---

但是我的问题其实更恶心。我是在test目录下还有一个文件d.py

```python
def fword():
    print("ffff")
```

而且我试图在foo.py中导入这个函数`from ...d import fword`并使用。

那么自然，如果要执行这个程序的话，直接`python path/to/foo.py`是肯定不行的。
我之前试图在test目录下`python -m mypackage/a/foo`也不行，并且提示`ValueError: attempted relative import beyond top-level package`。现在想想，确实很合理。因为执行这样的命令，就相当于把mypackage当成一个包了。那你在包里面import包外面的东西，显然不合理啊。

所以要解决这个，你得去test的parent目录下，执行`python -m test/mypackage/a/foo`。

ps：我之前用的解决方法是：用`ln -s`软链接了一个d.py到a目录。。。我真是太机智了。。。

---

到这里还没完，因为用`python -m`执行实在是太麻烦了。另外一个解决方案是用sys。
我在foo.py中加入这几行，并且把相对import改掉。
```python
import sys
sys.path.append("..")
sys.path.append("../..")

from spam import spam
from b.bar import bar
from d import fword
```
这样就可以执行了。但请注意⚠️，这样只能在a目录下才可以正常执行。原理很简单，就是增加了import的时候搜索的目录。这里是增加了父目录mypackage和test。所以import bar的时候还是要加上b。也正是因为如此，只能在a目录下执行这个代码。如果在test目录下执行`python mypackage/a/foo.py`的话，那就要改成
```python
sys.path.append('mypackage')
sys.path.append('.')
```

到这里应该就没有然后了。但是还是留下了一个问题，如果path里面有几个名字一样的文件，又会出现怎么样的表现？又该怎么办呢？这个我没试过。但是我觉得如果要用这个解决方案的话，肯定要尽量避免这种情况吧。




