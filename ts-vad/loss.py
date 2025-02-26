import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *

class Loss(nn.Module):
    def __init__(self, max_speaker):
        super(Loss, self).__init__()
        self.max_speaker = max_speaker
        self.fc = nn.Linear(int(4 / max_speaker * 96), 1)
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.m = nn.Sigmoid()

    def forward(self, x, labels=None): # x : B, max_speaker, T, int(4 / max_speaker * 96)   |     labels: B, max_speaker, T
        x = self.fc(x).squeeze(-1)        
        total_loss = 0

        for i in range(self.max_speaker):
            output = x[:, i, :]
            label = labels[:, i, :]
            loss = self.loss(output, label)
            total_loss += loss
        x = self.m(x)
        return total_loss / self.max_speaker, x