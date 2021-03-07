# 姑且记录一下吧

## clash

在ubuntu中使用clash，是在命令行中执行clash可执行文件，然后在浏览器 http://clash.razord.top/ 中进行配置。
clash命令行工具需要config.yaml和Country.mmdb这两个文件。
前一个是代理的配置，在clashX中有远程托管自动更新，但是在这里要定期自动更新，通过那个链接wget下来。
而后一个是全球ip之类的文件吧。因为在没有代理的时候自己下载太慢，所以可以看[这个链接](https://www.cnblogs.com/sueyyyy/p/12424178.html)下载。

在这之后，需要在系统的设置中设置代理。
目前的情况，基本就是http设置127.0.0.1 7890；而socks是7891。
另外就是在终端环境中代理命令，虽然之前谢过了，但是还是再写一遍：
```bash
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7891
```

## 桌面环境

安装了KDE-standard的桌面环境。之后需要选择桌面管理器，选了SDDM。
之后登陆的时候可以在左下角切换桌面环境。

## deb

- 不用root权限安装deb包（目前没有尝试过）：
```bash
dpkg -i package.deb --force-not-root --root=$HOME
```


