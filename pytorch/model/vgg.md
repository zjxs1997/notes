# VGG Net
VGG是14年提出来的工作，里面用的卷积全是3 * 3的，pooling用的都是2 * 2的maxpooling。
VGG有两个版本，16和19，区别其实就是19比16多了三个卷积层。

输入是一个(3, 224, 224)的image，输出的是对这个image的编码（大概）。

关于VGG的结构，其实看了这两个图就能明白了，不需要多说什么。
<div align="center">
    <img src="_img/vgg1.jpg" width="450">
</div>

<div align="center">
    <img src="_img/vgg2.jpg" width="450">
</div>
