### transformer的模板-v2.0

基于transformer目录下的模型改的。目标是把decoder的步骤改成增量式的，以节约时间。

是基于原来的代码，修改了model.py与model_decode.py这两个文件。

同样也没有经过测试，不知道能否跑起来。

另外就是，我在看原来的代码的时候，发现写的太烂了。如果有空（其实一直有空，只是懒而已），一定要重新写一遍transformer的代码。

新增：把原来的模型的decode方法换成这一套了，能跑起来了。但是只快了5秒，根本没什么卵用。

以及，发现了一个bug，那就是增量decode的时候的pos embedding，不能从0算起。