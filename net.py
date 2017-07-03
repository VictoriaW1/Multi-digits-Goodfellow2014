"""
Created on 2017.6.29 12:15
Author: Victoria
Email: wyvictoria1@gmail.com
"""
import torch.nn

class MultiDigitsNet(torch.nn.Module):
    def __init__(self):    
        super(MultiDigitsNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, stride=1, padding=2)
        self.conv5 = torch.nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.conv6 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.conv7 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.conv8 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2)
        
        self.fc9 = torch.nn.Linear(192*2*2, 3027)
        self.fc10 = torch.nn.Linear(3027, 3027)
        self.fc11_0 = torch.nn.Linear(3027, 7) #length
        self.fc11_1 = torch.nn.Linear(3027, 11) #digit1
        self.fc11_2 = torch.nn.Linear(3027, 11) #digit2
        self.fc11_3 = torch.nn.Linear(3027, 11) #digit3
        self.fc11_4 = torch.nn.Linear(3027, 11) #digit4
        self.fc11_5 = torch.nn.Linear(3027, 11) #digit5
        
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.relu = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()
    
    def forward(self, x):
        """
        The shape of input of net is 3*64*64.
        Input:
            x: input image, [batch_size, 3,, 64, 64]
        Return:
            len_probs: [batch_size, 7]
            digit1_probs: [batch_size, 10]
            digit2_probs: [batch_size, 10]
            digit3_probs: [batch_size, 10]
            digit4_probs: [batch_size, 10]
            digit5_probs: [batch_size, 10]               
        """
    
        x = self.max_pool1(self.relu(self.conv1(x)))
        x = self.max_pool2(self.relu(self.conv2(x)))
        x = self.max_pool1(self.relu(self.conv3(x)))
        x = self.max_pool2(self.relu(self.conv4(x)))
        x = self.max_pool1(self.relu(self.conv5(x)))
        x = self.max_pool2(self.relu(self.conv6(x)))      
        x = self.max_pool1(self.relu(self.conv7(x)))
        x = self.max_pool2(self.relu(self.conv8(x)))
        x = x.contiguous().view(-1, 192*2*2)
        x = self.relu(self.fc9(x))
        x = self.relu(self.fc10(x))
        len_probs = self.softmax(self.fc11_0(x))
        digit1_probs = self.softmax(self.fc11_1(x))
        digit2_probs = self.softmax(self.fc11_2(x))
        digit3_probs = self.softmax(self.fc11_3(x))
        digit4_probs = self.softmax(self.fc11_4(x))
        digit5_probs = self.softmax(self.fc11_5(x))
        
        return len_probs, digit1_probs, digit2_probs, digit3_probs, digit4_probs, digit5_probs
        
        
