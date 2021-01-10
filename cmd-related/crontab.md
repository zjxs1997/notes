# 定时任务

`crontab -l`查看当前以配置的定时任务。
`* * * * * echo 1 > ~/1.txt`

`crontab -r`删除当前用户的所有定时任务。

可以通过python-crontab这个库进行编辑。

```python
from crontab import CronTab
# 创建的定时任务只针对当前用户
cron_manager  = CronTab(user=True)
job = cron_manager.new(command='echo 1 > ~/1.txt')
# 设置每1min执行一次，具体语法其实可以看tldr的输出猜一下，
# 五个星号依次应该是分、时、日期、月份、星期
job.setall('*/1 * * * *')
# 还可以用job.hour.every(4)这种用法
cron_manager.write()
```