# clash for windows会稳定地每隔一段时间崩溃一下。整个重启脚本。
# 在task scheduler里可以设置定时任务
# windows默认不支持跑ps脚本，
# 执行方式：
# powershell.exe -executionpolicy bypass -File .\restart_clash_for_windows.ps1
# 也可以修改一下设置，用管理员权限运行Set-ExecutionPolicy Unrestricted。之后即可直接跑ps脚本了。

taskkill.exe /F /IM 'Clash for Windows.exe'
echo "clash exited"
& 'C:\Program Files\Clash for Windows\Clash for Windows.exe'
echo "clash restarted"

# 添加定时任务的时候，文件路径似乎要用双引号引起来。这是我在用ccleaner的时候想到的。
