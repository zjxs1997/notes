# clash for windows会稳定地每隔一段时间崩溃一下。整个重启脚本。
# 有没有那种可以设置每隔一段时间执行这个脚本的工具？
# 执行方式：
# powershell.exe -executionpolicy bypass -File .\restart_clash_for_windows.ps1

taskkill.exe /F /IM 'Clash for Windows.exe'
echo "clash exited"
& 'C:\Program Files\Clash for Windows\Clash for Windows.exe'
echo "clash restarted"

