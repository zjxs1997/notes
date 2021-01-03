几步走：

- 首先在远程服务器那里创建git repo：`git init --bare a.git`。

- 在本地git仓库关联过去：`git remote add origin rxs:~/a.git`（其中rxs是服务器的Host，后面是到git repo的路径）

- 本地push到远程，如果你xjb push的话，git会提示你，要使用命令：`git push --set-upstream origin master`。

- 在这之后应该都可以直接`git push origin`了。
