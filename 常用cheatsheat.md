记录一些觉得还比较有用，但是不太能记住或者不太熟悉的操作。如果觉得一个操作比较熟悉了，会注释掉。

### tmux
ctrl + b 之后：
<!-- - & ： 杀死windows -->
- % ： 在右边开panel
- " ： 在下面开panel
- [ ： 进入watch（？）模式，这个模式下 ctrl+s 可以搜索

### Linux

挂起进程： ctrl+z  
恢复挂起的进程： `fg % 进程号`  
<!-- 回到上一个位置：`cd -`   -->
绕过alias执行命令：`\ls`。服务器上的ls是`ls --color`的别名，有的时候有点碍事。  
重新执行上一个命令：`!!`，一般配合sudo用。

htop：
- 输入u可以查找用户的进程
- F9可以对某个进程发送信号，一般发9 SIGKILL杀了它
- / 可以搜索进程名



### vscode
<!-- - alt + ←/→：光标后退前进 -->
- ctrl + space： 触发编辑器提示
- ctrl + shift + t： 重新打开刚刚关闭的tab
- ctrl + shift + o： 跳转到某个符号
- ctrl + k, o： 在新窗口中打开当前文件。（在没有多窗口支持的现在，这个可以勉强算作替代方式吧）
<<<<<<< HEAD
- shift + alt/option + f：格式化代码，在cpp里比较有用，不过可以开自动化（设置editor.formatOnType或formatOnSave）

=======
- ctrl+k, ctrl+0: 折叠所有可以折叠的代码片段，在代码很长的时候很有帮助
>>>>>>> ce96916592e16634f838262881f2d346b4cf0023



