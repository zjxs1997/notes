# 这就是wsl的神奇之处

bug描述：wsl用ssh连接到集群服务器的时候，会不支持中文。具体表现为，在shell中敲入中文完全不显示，而在python中输出中文的时候直接报错。并且这样创建出来的shell的session创建的tmux也不支持中文。

分析：在终端里输入`echo $LANG`命令之后，会输出什么呢？很多系统会输出en_US.UTF-8，也有zh_CN.UTF-8，但是！wsl输出的是C.UTF-8，这就是wsl的神奇之处。就是因为这个，出现了上面描述的bug。

解决方法：
用`sudo vim /etc/default/locale`命令修改，把之前的`LANG=C.UTF-8`改为`LANG=zh_CN.UTF-8`。然后安装对应的语言包：`sudo apt-get install language-pack-zh-hans`，重启即可。


