import torch

class ReferentialLoss(Function):

    def forward(self, input, target):

        return loss   # a single number (averaged loss over batch samples)

    def backward(self, grad_output):
        ... # implementation
       return grad_input