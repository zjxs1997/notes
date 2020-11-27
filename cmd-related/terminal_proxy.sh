# 在bash等shell中设置代理的方法那是早就知道了，因为现在很多fq软件都会提供一个复制终端命令的选项。
# 大致就是，如果你http代理开在7890端口，socks5代理开在7891端口的话，就是执行这些命令
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7891

# Windows上的终端其实也可以做类似的事情，不过命令有点区别
# socks5的代理我导师不知道怎么设置【
set http_proxy=http://127.0.0.1:7890
set http_proxys=http://127.0.0.1:7890
