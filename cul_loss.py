import torch
import torch.nn as nn


class CustomLossWithRegularization(nn.Module):
    def __init__(self, lambda_reg):
        super(CustomLossWithRegularization, self).__init__()
        self.lambda_reg = lambda_reg
        self.bce_loss = nn.BCELoss()  # 二进制交叉熵损失

    def forward(self, y_pred, y_true, model):
        # 计算二进制交叉熵损失
        bce_loss = self.bce_loss(y_pred, y_true)

        # 计算L2正则化项
        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)  # 此处使用L2范数

        # 总损失等于二进制交叉熵损失加上L2正则化项
        total_loss = bce_loss + self.lambda_reg * l2_reg

        return total_loss
