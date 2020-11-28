#### 如何在服务器上跑jupyter

比如，我在icst4服务器上跑了个jupyter-lab --port=8889

如果我想在本地连接这个jupyter我得首先进行端口转发

`ssh -N -f -L localhost:8765:localhost:8889 icst4`
（注：icst4是ssh/config里的一个配置）

这之后，我在本地浏览器里输入`localhost:8765`就可以脸上jupyter-lab了。

---

为了方便起见，我在服务器配置了jpt脚本函数，直接执行`jpt <port>`即可。而在本地设置了`alias jptt='ssh -N -f -L localhost:8765:localhost:8889'`，因此在本地执行`jptt icst4`即可。
以及设置了
```bash
function jptf() {
    ssh -N -f -L localhost:$1:localhost:$2 $3
}
```
这个自由度更高的命令。可以`jptf 8765 8889 icst4`

但是需要**注意**，在zsh上，紧跟在参数后面的:l会被认为是参数的lower表示，因此在zsh中要把命令写成`ssh -N -f -L localhost:${1}:localhost:$2 $3`

但是我目前不知道如何关闭端口转发，就连关掉term也做不到。但貌似在切换代理/Wi-Fi等切换网络状态的时候就会自动关掉，神奇。

---

所以本质上就是一个端口转发的问题。学会了端口转发之后就可以做很多事情。比如在服务器上搞个code-server（vscode的服务端），然后本地端口转发，就可以用浏览器写代码了。不过并没有神什么卵用。。。


