# 添加右键菜单

众所周知，右键菜单打开terminal是正常操作系统都应该具备的一项功能。虽然windows没有，但是可以自己添加。

这里我添加了windows terminal（默认打开powershel）和wsl（ubuntu20.04）到右键菜单中。

只需要创建这两个reg文件：

```reg
Windows Registry Editor Version 5.00
 
[HKEY_CLASSES_ROOT\Directory\Background\shell\wt]
@="Windows Terminal Here"
"Icon"="C:\\Users\\15000\\AppData\\Local\\Windows Terminal\\wt.ico"
 
[HKEY_CLASSES_ROOT\Directory\Background\shell\wt\command]
@="C:\\Users\\15000\\AppData\\Local\\Microsoft\\WindowsApps\\wt.exe"
```

第二个：

```reg
Windows Registry Editor Version 5.00
 
[HKEY_CLASSES_ROOT\Directory\Background\shell\ubuntu]
@="Ubuntu Here"
"Icon"="C:\\Users\\15000\\AppData\\Local\\Windows Terminal\\ubuntu.ico"
 
[HKEY_CLASSES_ROOT\Directory\Background\shell\ubuntu\command]
@="C:\\Users\\15000\\AppData\\Local\\Microsoft\\WindowsApps\\wt.exe wsl"
```

然后双击这两个reg添加到注册表中即可。至于reg文件的语法，我也懒得搞明白，但是这段文件中的命令、icon位置这些应该还是能看懂的。啊对了，如果要把图片转换成ico格式的话，用`ffmpeg -i a.png a.ico`命令即可。

不过还有一点要注意，就是wt默认的setting.json的profile里面设置的startingDirectory都是"%USERPROFILE%"，也就是home目录。可以把相应的profile的这个项的值改成null，这样就可以在当前目录打开了。

效果图：

<div align="center">
    <img src="_img/demo.png" width="400">
</div>


