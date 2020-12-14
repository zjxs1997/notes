# ssh

## ~/.ssh/config

众所周知，在这个文件中可以写ssh配置文件。
比如说，写这么一个配置：
```
Host linux
User aaa
Hostname 123.4.5.6
```
那么在命令行中敲`ssh linux`的时候，其实就相当于在执行`ssh aaa@123.4.5.6`。
除了ssh之外，scp也可以用linux（只要服务器那边的.bashrc，或者别的文件里没有设置奇怪的东西）

配置里还可以指定私钥的目录`IdentityFile ~/.ssh/a.id_rsa`

如果ssh的目标是Windows的话，User这个字段要设置成设备名+\\（在配置里也得是两个反斜杠）+登陆用户名。

跳板🐔设置：
`ssh linux -J linux2`就是通过linux2作为跳板ssh到linux。需要多个跳板的话则以逗号隔开（真的有人会这么做吗。。。）`ssh linux -J l1,l2,l3`
跳板机的ssh版本需要在openssh 7.3以上。

跳板机在config中的设置也很简单，在Host下面加一行`ProxyJump linux2`即可。

可以参考[这个链接](https://wangdoc.com/ssh/client.html)

## 端口转发
关于端口转发，之前在[这篇笔记](../server-related/run-jupyter-on-server.md)也写过一些，但不是很熟悉。这里再总结以下。

端口转发分三种，分别是dynamic、local和remote。

### 动态转发

在本地执行命令`ssh -D local-port tunnel-host -N`。其中-D表示动态转发，-N表示只进行端口转发，不登陆；local-port为本地端口，tunnel-host为ssh服务器。

动态转发如图所示：
[动态转发](_img/dynamic_forward.png)

这种转发采用了socks5协议，访问外部网站时，需要把http请求转成socks5协议。

举个例子，连接校园网之后，执行`ssh -D 2121 rxs -N`，建立隧道。之后可以用命令`curl -x socks5://localhost:2121 http://172.31.32.100/gpustat/`获取网页信息（-x表示代理服务器）；或者在Firefox中设置代理为socks5，然后就可以访问这个看gpu使用状况的网页了。

### 本地转发

在本地执行`ssh -L local-port:target-host:target-port tunnel-host -N`。其中-L表示本地转发。target-host也可以写成localhost，表示是tunnel-host的localhost。这个应该算是比较熟悉了，之前用jupyter的[方法](../server-related/run-jupyter-on-server.md)都是这种本地转发。

此外说一句，-f表示的是在后台执行。用后台执行的话想直接退出就不是很方便，开个tmux更好。

本地转发如图所示：
[本地转发](_img/local_forward.png)

### 远程转发

这种场景比较特殊，本机在外网，跳板机和目标服务器都在内网，而且本机无法访问跳板机，跳板机可以访问本机。

在跳板机执行`ssh -R local-port:target-host:target-port -N local`，-R表示远程转发。需要本机安装了ssh server。

远程转发如图所示：
[远程转发](_img/remote_forward.png)


