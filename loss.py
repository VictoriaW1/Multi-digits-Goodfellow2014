"""
Created on 2017.7.3 8:25
Author: Victoria
Email: wyvictoria1@gmail.com
"""
import torch
from torch.autograd import Variable
def loss(output, target, cuda, gradcheck=False):
    """
    Compute loss between output of network and target.
    Input:
        output: (len_probs, digit1_probs, digit2_probs, digit3_probs, digit4_probs, digit5_probs), Float
        target: target of each image is [length, digit1, digit2, digit3, digit4, digit5], Byte
        cuda: 
        gradcheck: True: Double; False: Float
    Return:
        losses:    
    """
    losses = 0
    batch_size = output[0].size()[0]
    for i in range(6):
        classes_num = output[i].size()[1]
        one_hot_target = torch.zeros(batch_size, classes_num).scatter_(1, target[:, i].data.cpu().contiguous().view(batch_size, 1), 1) 
        one_hot_target = Variable(one_hot_target)
        if gradcheck:
            one_hot_target = one_hot_target.double()
        if cuda:
            one_hot_target = one_hot_target.cuda()
        cross_entropy_loss = -torch.sum(output[i].log() * one_hot_target, dim=1) #shape: (batch_size, )
        if i > 0:
            #Detecting whether digit in position i exists
            if gradcheck:
                cross_entropy_loss = torch.sum(cross_entropy_loss * (i <= target[:, 0]).double())
            else:
                cross_entropy_loss = torch.sum(cross_entropy_loss * (i <= target[:, 0]).float())    
        else:
            cross_entropy_loss = torch.sum(cross_entropy_loss)
        losses += cross_entropy_loss
    losses /= batch_size
    return losses
