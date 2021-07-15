import torch

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.6):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 

    def forward(self, inputs, targets):
        loss = self.alpha * (1.0-inputs)**self.gamma * (targets) * torch.log(inputs + 1e-14) + \
                (inputs)**self.gamma * (1.0 - targets) * torch.log(1.0 - inputs + 1e-14)
        loss = -torch.sum(torch.sum(loss, dim=-1), dim=-1)
        return loss


class HeatmapLoss(torch.nn.Module):
    def __init__(self, alpha=2, beta=4, reduction='mean'):
        super(HeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, inputs, targets):
        center_id = (targets == 1.0).float()
        other_id = (targets != 1.0).float()
        center_loss = -center_id * (1.0-inputs)**self.alpha * torch.log(inputs + 1e-14)
        other_loss = -other_id * (1 - targets)**self.beta * (inputs)**self.alpha * torch.log(1.0 - inputs + 1e-14)
        loss = center_loss + other_loss
        return loss