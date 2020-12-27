# 占位。。。

import argparse
parser = argparse.ArgumentParser()

# 添加参数，type为参数类型，如果命令行里没有指定，也没有default的话，则为None
parser.add_argument('--hello', type=str, default='world')
# choices参数的意思也很明显了，就是你只能在这里选
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'])
# 加了required之后必须在命令行中提供这个参数，否则报错
parser.add_argument('--shit', required=True)
# 设置一个flag，比如可以设置一个--debug的flag，--verbose的flag等
# 如果在命令行中加入--foo这个flag，那么args.foo的值就是True，否则false
parser.add_argument('--foo', action='store_true')

# 别的一些功能暂时还用不到，就写到这里【


args = parser.parse_args()
# 得到的args是一个namespace，可以把它当成一个object看。
print(args)

