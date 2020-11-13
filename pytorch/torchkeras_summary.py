import torch.nn as nn
from torchkeras import summary
# 依赖prettytable库

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 20)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, data):
        data = self.fc1(data)
        data = self.act1(data)
        data = self.fc2(data)
        return data

if __name__ == "__main__":
    m = Model()
    summary(m, input_shape=(100,))
# output
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Linear-1                   [-1, 20]           2,020
#               ReLU-2                   [-1, 20]               0
#             Linear-3                    [-1, 5]             105
# ================================================================
# Total params: 2,125
# Trainable params: 2,125
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.000381
# Forward/backward pass size (MB): 0.000343
# Params size (MB): 0.008106
# Estimated Total Size (MB): 0.008831
# ----------------------------------------------------------------

