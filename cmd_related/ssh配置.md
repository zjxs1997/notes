### ~/.ssh/config

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



