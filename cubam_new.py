import torch
import torch.nn as nn

class Cubam(nn.Module):
    def __init__(self, text_num, classifier_num):
        super().__init__()
        self.text_num = text_num
        self.classifier_num = classifier_num
        self.tau = nn.Parameter(torch.zeros(classifier_num, requires_grad=True), requires_grad=True)
        self.sigma = nn.Parameter(torch.ones(classifier_num, requires_grad=True) / 2, requires_grad=True)
        self.x = nn.Parameter(torch.zeros(text_num, requires_grad=True), requires_grad=True)
        self.normal = torch.distributions.Normal(0, 1)
        self.activate = nn.Tanh()
    
    def forward(self, L):
        self.sigma.data = self.sigma.data.clamp_(0.05, 200)
        loss = torch.tensor([0.0], device=0)
        negative_prob = self.normal.cdf(torch.stack([(self.tau - self.x[i]) / self.sigma for i in range(self.text_num)]))
        pi = torch.where(L != 1, negative_prob, 1.0 - negative_prob).clamp_(0.0001, 0.9999)
        loss -= torch.sum(torch.log(pi))
        loss -= torch.sum(torch.log(self.activate(2.5 - torch.abs(torch.clamp(self.tau, -2, 2)))))
        loss -= torch.sum(torch.log(self.activate(self.sigma * 4)))
        return loss