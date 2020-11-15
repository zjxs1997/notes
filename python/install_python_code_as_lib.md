写代码的时候总会感觉有些和以前写过的一样，再写一遍吧浪费时间，复制粘贴的话又觉得麻烦（？），不如把这些代码安装到python库里好了。

目录结构如下：

``` 
home/
|--pdemo
|  |--__init__.py
|  |--xxx.py
|--setup.py
```

setup.py中代码如下：

```python
from setuptools import find_packages, setup
setup(
  name='pdemo',
  version='0.0.1',
  packages=find_packages()
)
```

在home目录下面执行`python setup.py bdist_egg`，成功的话会出现build目录，dist目录以及egg-info，表示打包成功。然后再执行`python setup.py install`就可以安装了。