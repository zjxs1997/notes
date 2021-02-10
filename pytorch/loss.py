# 可以参考这个 《PyTorch的十八个损失函数》 https://zhuanlan.zhihu.com/p/61379965

# 待补充

import torch
import torch.nn as nn
import torch.optim as optim


# ==================================================================
# Focal Loss
# 是对binary cross entropy的改进，据说“在样本不均衡和存在较多易分类的样本时相比binary_crossentropy具有明显的优势”
# 它有两个可调参数，alpha参数和gamma参数。其中alpha参数主要用于衰减负样本的权重，gamma参数主要用于衰减容易训练样本的权重。
# 从而让模型更加聚焦在正样本和困难样本上。这就是为什么这个损失函数叫做Focal Loss。

class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0,alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self,y_pred,y_true):
        bce = torch.nn.BCELoss(reduction = "none")(y_pred,y_true)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        loss = torch.mean(alpha_factor * modulating_factor * bce)
        return loss

#困难样本
y_pred_hard = torch.tensor([[0.5],[0.5]])
y_true_hard = torch.tensor([[1.0],[0.0]])

#容易样本
y_pred_easy = torch.tensor([[0.9],[0.1]])
y_true_easy = torch.tensor([[1.0],[0.0]])

focal_loss = FocalLoss()
bce_loss = nn.BCELoss()

print("focal_loss(hard samples):", focal_loss(y_pred_hard,y_true_hard))
print("bce_loss(hard samples):", bce_loss(y_pred_hard,y_true_hard))
print("focal_loss(easy samples):", focal_loss(y_pred_easy,y_true_easy))
print("bce_loss(easy samples):", bce_loss(y_pred_easy,y_true_easy))

# 输出
# focal_loss(hard samples): tensor(0.0866)
# bce_loss(hard samples): tensor(0.6931)
# focal_loss(easy samples): tensor(0.0005)
# bce_loss(easy samples): tensor(0.1054)
#可见 focal_loss让容易样本的权重衰减到原来的 0.0005/0.1054 = 0.00474
#而让困难样本的权重只衰减到原来的 0.0866/0.6931=0.12496
# 因此相对而言，focal_loss可以衰减容易样本的权重。


# ==================================================================
# L1与L2正则项Loss

# L1正则化
def L1Loss(model,beta):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss +  beta * torch.sum(torch.abs(param))
    return l1_loss

# L2正则化
def L2Loss(model,alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name: #一般不对偏置项使用正则
            l2_loss = l2_loss + (0.5 * alpha * torch.sum(torch.pow(param, 2)))
    return l2_loss
# L2正则项其实可以在优化器里直接使用，不用自己代码实现：
model = nn.Linear(100, 50)
weight_params = [param for name, param in model.named_parameters() if "bias" not in name]
bias_params = [param for name, param in model.named_parameters() if "bias" in name]

optimizer = optim.SGD(
    [
        {'params': weight_params, 'weight_decay':1e-5},
        {'params': bias_params, 'weight_decay':0}
    ],
    lr=1e-2,
    momentum=0.9,
)


# ==================================================================
# KL散度loss
# kld接受预测的概率分布input和标准概率分布target作为输入，input和target的shape应当一致
# 需要注意，input必须是log softmax
# 而target需要是softmax，
# 但是在版本比较新的pytorch中，KLDivLoss可以指定参数log_target为True，在这种情况下，target也是log softmax
kld = nn.KLDivLoss()
kld_log = nn.KLDivLoss(log_target=True)
a = torch.log_softmax(torch.rand(1, 2), dim=-1)
b = torch.rand(1, 2)
print(kld(a, torch.softmax(b, dim=-1)))
print(kld_log(a, torch.log_softmax(b, dim=-1)))

# 输出：
# tensor(0.0114)
# tensor(0.0114)

# 根据kld计算公式，loss应当是a * log(a/a') = a * log(a) - a * log(a')
# 其中a是target，而a'是input的预测概率，因此：
bb = torch.softmax(b, dim=-1)
torch.sum(bb*torch.log(bb) - bb*a) / 2
# 输出：
# tensor(0.0114)

#也就是说自己计算和调用KLD计算应该是没有区别的，也就是说loss变成NaN这锅，Loss还是不背ORZ


