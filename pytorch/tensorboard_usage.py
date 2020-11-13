
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./data/tensorboard')
# writer里面有一些方法，虽然大部分都用不到吧=。=
# 可视化模型结构： writer.add_graph
# 可视化指标变化： writer.add_scalar
# 可视化参数分布： writer.add_histogram
# 可视化原始图像： writer.add_image 或 writer.add_images
# 可视化人工绘图： writer.add_figure

# 在用writer写了一些东西之后，可以在命令行中执行
# tensorboard --logdir ./data/tensorboard
# 然后会输出一些信息，在指定的端口就可以看到相应的可视化网页

# add_scalar方法是用来统计标量的，一般用来可视化loss曲线或者是metric上的表现
import random
for i in range(10):
    writer.add_scalar("x", random.random(), i)
    writer.add_scalar("y", random.random(), i)

# 如果需要对模型的参数(一般非标量)在训练过程中的变化进行可视化，可以使用 writer.add_histogram。
# 目前还没用过，不知道效果如何
for i in range(10):
    writer.add_histogram('weight', torch.randn((100, 30)), i)
    writer.flush()

# add_image系列估计现在也用不到吧，大概就是传入图片的tensor
# add_figure要传入matplotlib的figure对象



