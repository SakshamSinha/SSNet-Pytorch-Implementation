from torchvision import models
import torch.nn as nn
import torch
import pdb
import torch.nn.functional as F

class Demo_Model(nn.Module):
    def __init__(self, nClasses = 200):
        super(Demo_Model,self).__init__();
        
        self.conv_1 = nn.Conv2d(3,32,kernel_size=5,stride=1, padding = 2)
        self.relu_1 = nn.ReLU(True);
        self.batch_norm_1 = nn.BatchNorm2d(32);
        self.pool_1 = nn.MaxPool2d(kernel_size = 2, stride =2)
        
        self.conv_2 = nn.Conv2d(32,32,kernel_size=5,stride=1, padding = 2)
        self.relu_2 = nn.ReLU(True);
        self.batch_norm_2 = nn.BatchNorm2d(32);
        '''self.pool_2 = nn.MaxPool2d(kernel_size = 2, stride =2)

        self.conv_3 = nn.Conv2d(32,32,kernel_size=5,stride=1, padding = 2)
        self.relu_3 = nn.ReLU(True);
        self.batch_norm_3 = nn.BatchNorm2d(32);'''

        self.fc_1 = nn.Linear(32768, 1024);
        #self.fc_1 = nn.Linear(8192, 200);
        self.relu_4 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm1d(1024);
        self.dropout_1 = nn.Dropout(p = 0.5);
        self.fc_2 = nn.Linear(1024, nClasses);
        
    def forward(self,x):
        #pdb.set_trace();
        y = self.conv_1(x)
        y = self.relu_1(y)
        y = self.batch_norm_1(y)
        y = self.pool_1(y)
        
        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.batch_norm_2(y)
        '''y = self.pool_2(y)
        
        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_3(y)'''


        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)
        y = self.dropout_1(y)
        y = self.fc_2(y)
        return(y)

class SimpleConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(SimpleConv2dLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        return F.relu(x)

class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = SimpleConv2dLayer(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = SimpleConv2dLayer(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = SimpleConv2dLayer(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = SimpleConv2dLayer(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = SimpleConv2dLayer(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = SimpleConv2dLayer(96, 96, kernel_size=3, padding=1)

        self.branch_pool = SimpleConv2dLayer(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]

        return torch.cat(outputs, 1)

class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3_1 = SimpleConv2dLayer(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = SimpleConv2dLayer(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = SimpleConv2dLayer(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = SimpleConv2dLayer(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = SimpleConv2dLayer(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = SimpleConv2dLayer(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)

class SSNet_Model(nn.Module):
    def __init__(self, nClasses=200):
        super(SSNet_Model, self).__init__();

        #64x64x3
        self.conv2d_1_3x3= SimpleConv2dLayer(3, 32, kernel_size=3, stride=1, padding=1)

        #64x64x32
        self.conv2d_2_3x3 = SimpleConv2dLayer(32, 64, kernel_size=3, stride=1, padding=1)

        #64x64x64
        self.mixed_3 = InceptionA(64, pool_features=32)

        self.mixed_4 = InceptionB(256)
        self.fc = nn.Linear(768, nClasses)

        # self.conv2d_1_3x3 = SimpleConv2dLayer(256, 288, kernel_size=3, stride=1, padding=1)



        '''self.pool_2 = nn.MaxPool2d(kernel_size = 2, stride =2)

        self.conv_3 = nn.Conv2d(32,32,kernel_size=5,stride=1, padding = 2)
        self.relu_3 = nn.ReLU(True);
        self.batch_norm_3 = nn.BatchNorm2d(32);'''
        # #32x32x32
        # self.fc_1 = nn.Linear(32768, 1024); #32x32x32 (1-D), 1024 output signals
        # #
        # # self.fc_1 = nn.Linear(8192, 200);
        # self.relu_4 = nn.ReLU(True);
        # self.batch_norm_4 = nn.BatchNorm1d(1024);
        # self.dropout_1 = nn.Dropout(p=0.5);
        # self.fc_2 = nn.Linear(1024, nClasses); #1024 (1-D), num_classes signals

        #added weights initalisations with normal distributions and set biases to 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
            #m.bias.data.zero_()
            #m.bias.data.fill_(0)


    def forward(self, x):
        # pdb.set_trace();
        #64x64x3
        y = self.conv2d_1_3x3(x)
        #64x64x32
        y = self.conv2d_2_3x3(y)
        #64x64x64
        y = F.max_pool2d(y, kernel_size=3, stride=2)
        #32x32x64
        y = self.mixed_3(y)
        #32x32x256
        y = self.mixed_4(y)
        #15x15x768
        y = F.max_pool2d(y, kernel_size=15)
        #1x1x768
        y = F.dropout(y, training=self.training)
        y = y.view(y.size(0), -1)
        #768
        y = self.fc(y)
        return (y)


def resnet18(pretrained = True):
    return models.resnet18(pretrained)

def alexnet(pretrained = True):
    return models.alexnet(pretrained)
    
def vgg16(pretrained = True):
    return models.vgg16(pretrained)

def demo_model():
    return Demo_Model();

def ssnet_model():
    return SSNet_Model();
