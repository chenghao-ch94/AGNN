import torch.nn as nn

from .models import register

def conv_block(in_channels, out_channels):
   return nn.Sequential(
       nn.Conv2d(in_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),
       nn.ReLU(),
    #    nn.LeakyReLU(0.2,inplace=True),
       nn.MaxPool2d(2)
   )

def conv_block_n(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2, inplace=True)
    )
def conv_block_n2(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2, inplace=True)
    )
def conv_block_n3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout2d(0.4)
    )
def conv_block_n4(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout2d(0.5)
    )

@register('convnet4_128')
class ConvNet4_128(nn.Module):

   def __init__(self, x_dim=3, hid_dim=128, z_dim=128):
       super().__init__()
       self.layer1 = conv_block_n(x_dim, hid_dim)
       self.layer2 = conv_block_n2(hid_dim, int(hid_dim*1.5))
       self.layer3 = conv_block_n3(int(hid_dim*1.5), hid_dim*2)
       self.layer4 = conv_block_n4(hid_dim*2, hid_dim*4)

       self.fc = nn.Linear(hid_dim*4*5*5, z_dim, bias=True)
       self.fc_bn = nn.BatchNorm1d(z_dim)
       self.out_dim = z_dim


   def forward(self, x):
       x = self.layer1(x)                      
       x = self.layer2(x)     
       x = self.layer3(x)
       x = self.layer4(x)
       x_cls = self.fc_bn(self.fc(x.view(x.size(0),-1)))
       return x_cls


@register('convnet4')
class ConvNet4(nn.Module):

   def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
       super().__init__()
       self.layer1 = conv_block(x_dim, hid_dim)
       self.layer2 = conv_block(hid_dim, hid_dim*2)
       self.layer3 = conv_block(hid_dim*2, hid_dim*4)
       self.layer4 = conv_block(hid_dim*4, z_dim*4)
       self.avgpool = nn.AvgPool2d(5, stride=1)
       self.fc = nn.Linear(256, 128, bias=True)
       self.fc_bn = nn.BatchNorm1d(128)
       self.out_dim = 128


   def forward(self, x):
       x = self.layer1(x)                      
       x_sim = self.layer2(x)     
       x = self.layer3(x_sim)
       x_cls = self.layer4(x)
       x_cls = self.fc_bn(self.fc(self.avgpool(x_cls).squeeze()))
       return x_cls