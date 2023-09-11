import torch
import torch.nn as nn

# class Clamp(torch.autograd.Function):
#     def __init__(self, min, max):
#         super().__init__()
#         self.min = min
#         self.max = max

#     @staticmethod
#     def forward(ctx, input):
#         return input.clamp(min=self.min, max=self.max) # the value in iterative = 2

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.clone()

# clamp_1 = Clamp(-1, 1)
# clamp_2 = Clamp(0.1, 10)
# module
class Cubam(nn.Module):
    def __init__(self, text_num, classifier_num, sigma_z = 0.8):
        super().__init__()
        self.text_num = text_num
        self.classifier_num = classifier_num
        self.tau = nn.Parameter(torch.zeros(classifier_num, requires_grad=True), requires_grad=True)
        self.sigma = nn.Parameter(torch.ones(classifier_num, requires_grad=True) / 2, requires_grad=True)
        self.x = nn.Parameter(torch.zeros(text_num, requires_grad=True), requires_grad=True)
        self.normal = torch.distributions.Normal(0, 1)
        self.sigma_z = sigma_z
        self.activate = nn.Tanh()
        # self.dropout = nn.Dropout(0.01)
    
    def forward(self, L):
        self.sigma.data = self.sigma.data.clamp_(0.05, 200)
        # self.tau.data = self.tau.data.clamp_(-2, 2)
        loss = torch.tensor([0.0], device=0)
        negative_prob = self.normal.cdf(torch.stack([(self.tau - self.x[i]) / self.sigma for i in range(self.text_num)]))
        pi = torch.where(L != 1, negative_prob, 1.0 - negative_prob).clamp_(0.0001, 0.9999)
        loss -= torch.sum(torch.log(pi))
        loss -= torch.sum(torch.log(self.activate(2.5 - torch.abs(torch.clamp(self.tau, -2, 2)))))
        loss -= torch.sum(torch.log(self.activate(self.sigma * 4)))
        # loss -= torch.sum(torch.where(self.sigma < 0.3, torch.log(self.sigma * 2), torch.tensor([0], device=0)))
        return loss