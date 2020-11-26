# 控制终端颜色显示

from colorama import Back, Fore, Style

print(f"hello {Back.GREEN}{Fore.RED} world, {Style.RESET_ALL} man.")


# 另外有个rich库，似乎也可以起到类似的作用
