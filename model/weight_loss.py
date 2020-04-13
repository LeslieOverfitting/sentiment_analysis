import torch
import torch.nn as nn
import torch.nn.functional as func

class myLoss(nn.Module):
    def __init__(self,parameters)
        self.params = self.parameters

    def forward(self, predict, label):
      """
        predict ： batch_size * 3
        label: batch_size * 1
      “”“
        loss = cal_loss(self.params)
        return loss
