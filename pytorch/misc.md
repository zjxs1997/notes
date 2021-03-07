# 各种不方便归类的乱七八糟的操作

## 固定模块中某一子模块的参数

只需要设定该模块的requires_grad为False即可，或者直接调用方法`m.requires_grad_(False)`

