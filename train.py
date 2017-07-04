"""
Created on 2017.6.29 15:15
Author: Victoria
Email: wyvictoria1@gmail.com
"""
import time
import gc
import matplotlib.pyplot as plt
from load_data import generate_dataset, SVHNDataset
from net import MultiDigitsNet
from loss import loss
from accuracy import accu

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as init
            
def train(data_aug, rand_num, batch_size, epochs, lr, momentum, early_stopping, log_interval, cuda, path):
    """
    Training model with data provided.
    Input:
        data_aug:
        rand_num:
        batch_size:
        epochs:
        lr:
        momentum:
        log_interval: how many batches to wait before logging training status.
        early_stopping: patient iters
        cuda: 
        path: path to save trained model
    """
    print "training..."
    print "cuda: ", cuda
    torch.manual_seed(1)
    if cuda:
        torch.cuda.manual_seed(1)
        
    train_data = generate_dataset(train="train", data_aug=data_aug, rand_num=rand_num)
    split = int(len(train_data) * 0.8)
    training_data = (train_data[0][0:split], train_data[1][0:split])
    dev_data = (train_data[0][split:], train_data[1][split:])
    print "len of training image: ", len(training_data[0])
    print "len of dev images: ", len(dev_data[0])
    train_dataset = SVHNDataset(training_data[0], training_data[1]) 
    kwargs = {'num_workers': 1, 'pin_memory': True}  if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)                
    dev_dataset = SVHNDataset(dev_data[0], dev_data[1])
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, **kwargs)
    
    model = MultiDigitsNet() 
    model.apply(weights_init)   
    model.train()   
    if cuda:
        model.cuda() #save all params to GPU
        
    #optimizer = optim.Adagrad(self.parameters(), lr=lr)#self.parameters() 
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  
    loss_history = [] 
    accuracy_history = [] 
    dev_loss_history = []
    dev_accuracy_history = []
    best_dev_loss = float("inf")
    best_model = model.state_dict()  
    patient = 0 
    for epoch in range(epochs):
        train_loss = 0
        train_accuracy = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            print lr
            data, target = Variable(data), Variable(target)
            if cuda:
                data, target = data.cuda(), target.cuda()
            data = data.float()    
            optimizer.zero_grad()
            output = model(data)
            losses = loss(output, target, cuda) 
            train_loss += losses.data[0]
            losses.backward()
            #print model.conv1.weight.data[0, 0]
            #print model.conv1.weight.grad.data[0, 0]    
            optimizer.step()           
            accuracy = accu(output, target, cuda)
            train_accuracy += accuracy.data[0]
            if batch_idx % log_interval == 0:
                print "epoch: {} [{}/{}], loss: {}, accuracy: {}".format(epoch, batch_idx*len(data), len(train_loader.dataset), losses.data[0], accuracy.data[0])
            gc.collect()    
        loss_history.append(train_loss / len(train_loader))
        accuracy_history.append(train_accuracy / len(train_loader))
        if (epoch+1)%30==0:
            lr = lr * 0.9
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            
        #early stopping
        dev_loss = 0
        dev_accuracy = 0
        for batch_idx, (dev_data, dev_target) in enumerate(dev_loader):
            if cuda:
                dev_data, dev_target = dev_data.cuda(), dev_target.cuda()
            dev_data, dev_target = Variable(dev_data), Variable(dev_target)
            dev_data = dev_data.float()    
            dev_output = model(dev_data)           
            dev_loss += loss(dev_output, dev_target, cuda).data[0]
            dev_accuracy += accu(dev_output, dev_target, cuda).data[0]
        dev_loss_history.append(dev_loss / len(dev_loader))
        dev_accuracy_history.append(dev_accuracy / len(dev_loader))     
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_model = model.state_dict()
            patient = 0
        else:
            print "dev_loss not decrease: best_loss: {}, dev_loss: {}".format(best_dev_loss, dev_loss)
            patient += 1
            if patient > early_stopping:
                break      
    torch.save(best_model, path)     
    plt.figure()
    plt.plot(range(len(loss_history)), loss_history)  
    plt.savefig("figure/train_losses.png") 
    plt.figure()
    plt.plot(range(len(dev_loss_history)), dev_loss_history)
    plt.savefig("figure/dev_losses.png") 
        
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        init.kaiming_uniform(m.weight)
        init.constant(m.bias, 0.01)
    if isinstance(m, torch.nn.Linear):
        init.xavier_uniform(m.weight)
        init.constant(m.bias, 0.01)     
        
if __name__=="__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    train_data = generate_dataset(train="test", data_aug=False)
    print "len of image: ", len(train_data[0])
    train_dataset = SVHNDataset(train_data[0], train_data[1]) 
    kwargs = {'num_workers': 1, 'pin_memory': True}  
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, **kwargs)                

    model = MultiDigitsNet()
    model.cuda()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        model.double()
        data = data.double()
        grad_check(model, data, target, cuda=True)    
        
