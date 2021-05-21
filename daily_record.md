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

## 2021.4.7
python的几个命令行参数
- `-m`，执行某个模块的代码
- `-i`，进入可交互界面执行某脚本（如果脚本出错，也会保留在交互式界面中）
- `-q`，静默模式，进入交互式界面前不输出文本
- `-c`，在shell中输入python脚本执行

## 2021.4.15
clash的订阅链接，相当于是返回一个txt文件，txt文件里写的是各种配置，包括tz群组、规则等，当然还有DNS相关的。
群组可以递归地返回一个订阅链接，然后那个链接里面才是具体的tz配置内容。这些东西最后都是存在本地的，可以修改本地的文本来改变配置。
比如最近几天遇到的问题是DNS相关的，没法访问订阅链接返回的DNS，只要在配置里面删除DNS相关的那一坨就行了。

## 2021.5.14
python有个`statistics`库，这个库中有一些可以用来计算统计量的函数。比如：
```python
statistics.mean([1,2,3,4])
# 2.5

statistics.stdev([1,2,3,4])
# 1.2909944487358056
# = sqrt(5/3)
```


