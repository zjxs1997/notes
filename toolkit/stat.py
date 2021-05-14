# 计算均值、方差、协方差、pearson系数
# pearson，就是协方差除以两个方差之积
import statistics

X = [1, 2, 3, 4]
Y = [4, 3, 2, 1]

x_avg = statistics.mean(X)
y_avg = statistics.mean(Y)
x_std = statistics.stdev(X)
y_std = statistics.stdev(Y)

cov = 0
for x, y in zip(X, Y):
    cov += (x - x_avg) * (y - y_avg)
cov = cov / (len(X)-1)

pearson = cov / (x_std * y_std)
print(pearson)

# pearson系数其实可以用scipy直接计算：
from scipy.stats import pearsonr
pearsonr([1,2,3,4], [4,3,2,1])
# Out: (-1.0, 0.0)

# spearman系数的计算感觉比较复杂
# 计算公式这里也不写了，直接用scipy的库
from scipy.stats import spearmanr
spearmanr([1,2,3,4], [4, 3, 2, 1])
# Out: SpearmanrResult(correlation=-1.0, pvalue=0.0)


