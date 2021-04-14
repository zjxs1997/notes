powershell默认的补全功能非常辣鸡，但是其实可以通过一些配置修改做的还不错。

首先看`$PROFILE`这个变量，一般来讲是在`~\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1`这个路径，这个文件相当于bash的`.bashrc`。

创建一个配置文件，然后在里面写入：
```ps1
# Shows navigable menu of all options when hitting Tab
Set-PSReadlineKeyHandler -Key Tab -Function MenuComplete

# Autocompletion for arrow keys
Set-PSReadlineKeyHandler -Key UpArrow -Function HistorySearchBackward
Set-PSReadlineKeyHandler -Key DownArrow -Function HistorySearchForward
```

这几行配置的意思太明显了吧，第一行是类似于bash风格的弹出菜单，可以看到所有补全选项。第二个是类似于zsh风格的，用上下键选择匹配历史中类似的命令。

和`bash`之类的一样，也可以在这里创建别名：

```ps1
set-alias echo Write-Ouput
# 这个似乎是默认的别名，rm、mv之类的别名也有
```

创建函数：
```ps1
Function scpi
{
    scp $args[0] i1:/specific/path
}
```
