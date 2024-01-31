"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute


Code adapted from CMSC733 at the University of Maryland, College Park.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    criteria  = nn.CrossEntropyLoss()
    loss = criteria(out,labels)
    return loss

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        #print(out.size())
        loss = loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))




class CIFAR10Model(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      super().__init__()
      
      
      #############################
      self.conv1 = nn.Conv2d(3,6,5)
      self.pool = nn.MaxPool2d(2,2)
      self.conv2 = nn.Conv2d(6,16,5)
     
      self.fc1 = nn.Linear(16*5*5, 120)
      self.fc2 = nn.Linear(120,84)
      self.fc3 = nn.Linear(84,OutputSize)
      
  def forward(self, xb):
      
      out = self.pool(F.relu(self.conv1(xb)))
    #   print(out.shape)
      out = self.pool(F.relu(self.conv2(out)))
    #   print(out.shape)
      out = out.view(-1, 16*5*5)
    #   print(out.shape)
      out = F.relu(self.fc1(out))
      out = F.relu(self.fc2(out))
      out = self.fc3(out)
      
      return out


class ResidualBlock(nn.Module):
    #this is residual block of the network used in ResNet
    def __init__(self, in_channels, out_channels, stride = 1,downsample = None):
        expansion =1 
        #print("###########################################################")
        #print(in_channels)
        #print(out_channels)
        super().__init__()
        
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))

        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

        

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        
        #print(x.size())
        #print(residual.size())
        #print("*&&&&&&&&&&&&&&&&&&&&&&&7")
        x += residual
        out = self.relu(x)
        return out
        

class ResNet(ImageClassificationBase):
    def __init__(self, block,no_blocks, no_classes,in_channels):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, self.in_planes, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(self.in_planes),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.create_layer(block, 64, no_blocks[0],stride = 1)
        self.layer2 = self.create_layer(block, 128, no_blocks[1], stride = 2)
        self.layer3 = self.create_layer(block, 256, no_blocks[2], stride = 2)
        self.layer4 = self.create_layer(block, 512, no_blocks[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear1 = nn.Linear(8192, no_classes)

    def create_layer(self, block, no_planes, no_blocks, stride):
        downsample = None
        #stride != 1 or self.in_planes != no_planes
        if self.in_planes != no_planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, no_planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(no_planes),
            )
        layers = []
        layers.append(block(self.in_planes, no_planes, stride, downsample))
        self.in_planes = no_planes
        for i in range(1, no_blocks):
            layers.append(block(self.in_planes, no_planes))

        return nn.Sequential(*layers)  
    def forward(self, xb):
        x = self.conv1(xb)
        x = self.layer1(x) ##create resnet blocks layers
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x

class ResNextBlock(nn.Module):
    def __init__(self, in_channels, cardinality, bottleneck_width, downsample=None, stride = 1 ):
        super().__init__()
        self.expansion =2
        out_channels = cardinality * bottleneck_width
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3,groups=cardinality, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())

        self.conv3 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size = 1, stride = 1, padding = 0),
                        nn.BatchNorm2d(out_channels * self.expansion))

        self.downsample = downsample
        self.relu = nn.ReLU()                 

    def forward(self, x):
        residual = x
        #print(x.size())
        if self.downsample is not None:
            residual = self.downsample(x)
            #print("Here")
        #print(residual.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #print(x.size())
        x += residual
        out = self.relu(x)
        return out





class ResNext(ImageClassificationBase):
    def __init__(self, block,no_blocks,cardinality, bottleneck_width ,no_classes,in_channels):
        super().__init__()
        self.in_planes = 64
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, self.in_planes, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(self.in_planes),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self.create_layer(block, no_blocks[0],stride = 1)
        self.layer2 = self.create_layer(block, no_blocks[1], stride = 2)
        self.layer3 = self.create_layer(block, no_blocks[2], stride = 2)
        self.layer4 = self.create_layer(block, no_blocks[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear1 = nn.Linear(8192, no_classes)

    def create_layer(self, block, no_blocks, stride):
        downsample = None
        no_planes = self.cardinality * self.bottleneck_width ###output channels
        
        if stride != 1 or self.in_planes != no_planes *2:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, no_planes *2, kernel_size=1, stride=stride),
                nn.BatchNorm2d(no_planes * 2),
            )
        layers = []
        #print("Input")
        layers.append(block(self.in_planes, self.cardinality, self.bottleneck_width,  downsample, stride))
        self.in_planes = no_planes *2 
        for i in range(no_blocks -1):
            layers.append(block(self.in_planes, self.cardinality, self.bottleneck_width))
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    


    def forward(self, xb):
        x = self.conv1(xb)
        #x = self.maxpool(x)
        x = self.layer1(x) ##create resnet blocks layers
        #print(x.size())
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x

class BottleNeck(nn.Module):
    """ This is a bottleneck class for the DenseNet implementation
        this is basic building bloack of DenseNet
    """
    def __init__(self, in_channels,growth_rate ):
        super().__init__()
        ##define layers of the block
        print(in_channels)
        self.conv1 = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, 4 * growth_rate, kernel_size = 1, bias = False) 
        )
    
        self.conv2 = nn.Sequential(
        nn.BatchNorm2d(4 * growth_rate),
        nn.Conv2d(4 * growth_rate, growth_rate, kernel_size = 3,padding= 1, bias = False),
        nn.ReLU() 
        )

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        print("Bottleneck output size before concat:", out.size())
        x = torch.cat([out,x], 1)
        print("Bottleneck output size after concat:", x.size())
        return x

class Transition(nn.Module):    
    """
    This is transition class, used for transition between multiple layers of DenseNet
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.avgpool = nn.AvgPool2d(2)

    def forward(self,x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        out = self.avgpool(x)
        return out
        
class DenseNet(ImageClassificationBase):
    def __init__(self, block, no_blocks,  no_classes,in_channels, growth_rate = 12, reduction = 0.5):
        super().__init__()
        self.growth_rate = growth_rate
        self.no_planes = 2 * growth_rate

        #first convolution layer - 3 to 64
        self.conv1 = nn.Conv2d(in_channels, self.no_planes, kernel_size =3, padding = 1, bias = False)

        #create first dense layer
        self.dense1 = self.create_layer(block, self.no_planes, no_blocks[0])
        self.no_planes += no_blocks[0] * growth_rate
        self.out_channels = int(math.floor(self.no_planes * reduction)) 
        self.trans1 = Transition(self.no_planes, self.out_channels)
        self.no_planes = self.out_channels

        #dense layer 2
        self.dense2 = self.create_layer(block, self.no_planes, no_blocks[1])
        self.no_planes += no_blocks[1] * growth_rate
        self.out_channels = int(math.floor(self.no_planes * reduction)) 
        self.trans2 = Transition(self.no_planes, self.out_channels)
        self.no_planes = self.out_channels

        #dense layer 3
        self.dense3 = self.create_layer(block, self.no_planes, no_blocks[2])
        self.no_planes += no_blocks[2] * growth_rate
        self.batch_norm1 = nn.BatchNorm2d(self.no_planes)
        final_size = 32
        final_size = final_size // 2  
        final_size = final_size // 2 
        final_size = final_size // 4 
        self.final_num_features = self.no_planes * final_size * final_size
        self.linear1 = nn.Linear(self.final_num_features, no_classes)



    def create_layer(self,block, no_planes, no_blocks):
        print("Here###################3")
        print(no_planes)
        layers = []
        for i in range(no_blocks):
            layers.append(block(no_planes, self.growth_rate))
            no_planes += self.growth_rate
        out = nn.Sequential(*layers)

        return out

    def forward(self, xb):
        x = self.conv1(xb)
        print("Initial conv output size:", x.size())
        x = self.dense1(x)
        x = self.trans1(x)

        x = self.dense2(x)
        x = self.trans2(x)

        x = self.dense3(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 4)
        
        x = x.view(x.size(0), -1) 
        assert x.size(1) == self.final_num_features
        out = self.linear1(x)

        return out