# 每日记录

不知道归类到哪或者觉得没必要专门开一个文件，但是想记一下的，就放到这好了

## 2021.1.4

- [CodeBert](https://github.com/microsoft/CodeBERT)，虽然说这个可以写到pytorch的transformers里面😅

## 2021.1.11

- sshfs，可以把目标系统的某个目录挂载到当前目录下面，有目标系统的ssh访问账号即可。
具体用法？看tldr。

## 2021.1.17

- 有个叫code_server的玩意，就是vscode的server版本。可以放在服务器上跑，然后开个端口。通过浏览器访问该端口，就可以看到一个网页版vscode的界面。
不过说实话并不使用，而且功能很大程度上被vscode的远程服务替代了。

## 2021.2.8

- python库pretty_errors，可以用彩色易读的方式展示报错信息。只要在终端环境中先运行`python -m pretty_errors`即可。但是感觉可能并没有太大的用处，因为可以用ipdb之类的代替。

## 2021.2.16

- python3似乎取消了sorted函数中的cmp参数，只支持key参数了。cmp是c里比较经典的一个用法，在python3中可以用替代品
```python
def cmp(a, b):
    pass
from functools import cmp_to_key
sorted(l, key=cmp_to_key(cmp))
```
