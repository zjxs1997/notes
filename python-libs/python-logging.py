# 新技巧！

# 可以通过`logging.Logger.manager.loggerDict`来获取已经被实例化的所有logger。  
# 通过这个dict，就可以很方便地做一些事情，比如huggingface的transformer库总喜欢log出一堆内容来，看起来有点烦。可以通过这段脚本禁用这些log：
# m = logging.Logger.manager
# for val in m.loggerDict.values():
#     if isinstance(val, logging.Logger):
#         val.setLevel(logging.ERROR)
# 不过要注意这段脚本的位置，需要在`import transformers`库之后执行。


import logging


# ==================== logging基本

# 不创建logger的话默认用root logger，直接打印到STDOUT

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(name)s:%(levelname)s:%(message)s')

logging.debug("man")
logging.info("shit")
logging.warning("woman")
logging.error("cool")
logging.critical("nice")

# print(logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)


# ==================== 创建logger

logger = logging.getLogger('test')
logger.setLevel(logging.DEBUG)

# 是否传播这个日志到祖先logger, 如果设置了False 就不会传到root logger(祖先Logger)的
# 默认StreamHandler那里， 也就是不会打印在页面上
logger.propagate = False

# 设置这个handler的处理格式， 实例化一个Formatter对象
apps_formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

# 添加handler, 决定日志落地到哪里，可以多个
# 这个是记录在文件的Handler，默认应该是append
apps_handler = logging.FileHandler(filename="apps.log")
apps_handler.setFormatter(apps_formatter)
logger.addHandler(apps_handler)
# 如果添加下面两行，则会同时打印到stdout，不同的handler甚至可以用不同的formatter
test_handler = logging.StreamHandler()
logger.addHandler(test_handler)

logger.warning("ohmygosh")


# ==================== 设置颜色可以用第三方库coloredlogs

import coloredlogs
# 对root logger操作
coloredlogs.install(level='DEBUG')
# 对某个logger操作
coloredlogs.install(logger=logger, level='INFO')

# 当然也可以纯手打，对五个等级分别设置不同的formatter，用控制字符操作颜色，logging库中好像没有默认的支持



def create_logger(level, name='myapp'):
    level = {
        'd': logging.DEBUG,
        "i": logging.INFO,
        'w': logging.WARNING,
        'e': logging.ERROR,
        'c': logging.CRITICAL,
    }[level]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    apps_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

    apps_handler = logging.FileHandler(filename=name+".log")
    apps_handler.setFormatter(apps_formatter)
    logger.addHandler(apps_handler)

    test_handler = logging.StreamHandler()
    test_handler.setFormatter(apps_formatter)
    logger.addHandler(test_handler)
    
    coloredlogs.install(logger=logger, level=level)

    return logger


