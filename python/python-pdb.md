### python调试工具pdb

pdb有点类似gdb，是一个命令行调试工具。用`pip`可以安装。有个ipdb库，与pdb的关系类似于ipython与python，是我比较常用的。

#### 基础使用

##### 在命令行里直接使用

比如要调试run.py文件，直接在命令行里`python -m ipdb run.py`即可，注意这种方式似乎没法`import sys`，会提示要先`import sys`再`import pdb`。

##### 在文件里使用

比如想调试run.py文件，锁定运行到某一行的时候会出bug。那可以在这一行之前加入代码：

```python
import ipdb
ipdb.set_trace()
```

效果和`python -m ipdb run.py`，然后在这一行加了断点一样，Python运行到这里后会自动停下来，然后就可以调试了。

还有这种用法：

```python
try:
  function(...)
 except xxx as e:
  __import__('pdb').post_mortem()
```

这样会回到抛出异常的地方。

这种写法其实很有用。有的时候跑一个程序不知道哪里会不会出个错之类的，可以用这个外面套一层。如果跑了很久，结果在最后的时候出了错退出，重跑要浪费很多时间。如果加了这个就可以进入pdb为所欲为了。——2020.11.06总结

##### 在别的软件使用

比如vscode和pycharm。

在ipython里也可以直接使用。执行一个代码块，或者`run run.py`出错的时候，可以直接输入`debug` magic命令，进入pdb调试现场，自动定位到报错的地方。当然jupyter notebook肯定也可以。



#### 调试命令

`l`：显示当前行前后的11行

`s`：step，执行，在第一个可能的地方（比如说，不是注释的地方，？）

`c`：continue，一直执行到断点（？

`n`：next，执行下一条语句

`b`：break，设置断点（用行号）。也可以设置到别的文件`b <filepath>:<line number>`，也可以设置到库文件里。如果不加任何参数，则会打印所有断点信息。

`cl`：clear，删除断点。参数是断点编号。

`p`：print，打印变量、表达式。命令`pp`，用pprint打印。`p locals()`打印所有本地变量。

`r`：return，执行到当前函数返回

`q`：quit。

`u`：up，跳到上一个frame。比如现在在a函数调用的b函数调用的c函数中出错了，想回到b函数看各种变量的值，就可以用u跳到b函数的调用堆栈。

`d`：down，与上一个相对。

**善用`h`：help命令。**







