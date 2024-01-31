#!/usr/bin/env python3

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

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)


import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
# import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import time
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
#from tqdm.notebook import tqdm
#import Misc.ImageUtils as iu
from Network.Network import CIFAR10Model
from Network.Network import ResidualBlock
from Network.Network import ResNet
from Network.Network import ResNextBlock
from Network.Network import ResNext

from Network.Network import BottleNeck
from Network.Network import DenseNet

from Misc.MiscUtils import *
from Misc.DataUtils import *
import torch.optim as optim
import wandb
from torchvision.transforms import v2
# Free up GPU memory
torch.cuda.empty_cache()

# Check GPU memory statistics
print(torch.cuda.memory_stats())

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
# Don't generate pyc codes
sys.dont_write_bytecode = True

    
def GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs: 
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch
   
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainSet)-1)
        
        ImageNum += 1
        I1, Label = TrainSet[RandIdx]      
        
          ##########################################################
          # Add any standardization or data augmentation here!
          ##########################################################
        
        transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = transforms(I1)


        

        # Append All Images and Mask
        I1Batch.append(img)
        LabelBatch.append(torch.tensor(Label))
        
      
    return torch.stack(I1Batch), torch.stack(LabelBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    #Prints all stats with all arguments
    """
    #print('Number of Epochs Training will run for ' + str(NumEpochs))
    #print('Factor of reduction in training data is ' + str(DivTrain))
    #print('Mini Batch Size ' + str(MiniBatchSize))
    #print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    

def TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, TrainSet, LogsPath):
    """
    Inputs: 
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    TrainSet - The training dataset
    LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath

    """
    # Initialize the model
    #model = CIFAR10Model(InputSize=3*32*32,OutputSize=10) 
    model = ResNet(ResidualBlock, [2, 2, 2,2],10,3)
    #model = ResNext(ResNextBlock, [3, 3, 3,3],2,64,10,3)
    model = DenseNet(BottleNeck, [4,8,12],10,3,32,0.5 )
    learning_rate = 1e-4 #how fast network paramters are updated
    ###############################################
    # Fill your optimizer of choice here!
    ###############################################

    model = model.to(device)
    Optimizer = optim.Adam(params = model.parameters(), lr = learning_rate, 
                                   betas=(0.5, 0.999))
    
    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + '.ckpt')
        # Extract only numbers from the name
        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')

    ##init Wandb params
    wandb.init(
            project="CNN_img_classification",

            config={
            
            }
            )
        
    for Epochs in range(StartEpoch, NumEpochs):
        #print(NumEpochs)
        #print(NumTrainSamples)
        #print(MiniBatchSize)
        #print(DivTrain)
        NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
               
        for PerEpochCounter in range(NumIterationsPerEpoch):
            Batch = GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize)

            #print("**********************************8")
            #print(Batch.size())
            # Predict output with forward pass
            LossThisBatch = model.training_step(Batch)
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()
            
            # Save checkpoint every some SaveCheckPoint's iterations
            #if PerEpochCounter % 100 == 0:
            #    # Save the Model learnt in this epoch
            #    print("Here#########################3")
            #    print(CheckPointPath)
            #    SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
            #    
            #    torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
            #    print('\n' + SaveName + ' Model Saved...')

            result = model.validation_step(Batch)
            model.epoch_end(Epochs*NumIterationsPerEpoch + PerEpochCounter, result)
            # Tensorboard
            Writer.add_scalar('LossEveryIter', result["loss"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            Writer.add_scalar('Accuracy', result["acc"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()
            wandb.log({"LossPerIteration": result["loss"]})
            wandb.log({"AccuracyPerIteration": result["acc"]})


            
        
            

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
        torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
        #print('\n' + SaveName + ' Model Saved...')
        wandb.log({"LossPerEpoch": result["loss"]})
        wandb.log({"AccuracyPerEpoch": result["acc"]})
        

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    #Parser = argparse.ArgumentParser()
    #Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    #Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    #Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    #Parser.add_argument('--MiniBatchSize', type=int, default=1, help='Size of the MiniBatch to use, Default:1')
    #Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    #Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
    TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=ToTensor())

    #Args = Parser.parse_args()
    #NumEpochs = Args.NumEpochs
    #DivTrain = float(Args.DivTrain)
    #MiniBatchSize = Args.MiniBatchSize
    #LoadCheckPoint = Args.LoadCheckPoint
    #CheckPointPath = Args.CheckPointPath
    #LogsPath = Args.LogsPath


    # Default Hyperparameters
    NumEpochs = 15
    #transforms = tf.Compose([ tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ResNext
    TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform= ToTensor())
    DivTrain = 1.0
    MiniBatchSize = 64
    LoadCheckPoint = 0
    CheckPointPath = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase2/Code/model/"
    LogsPath = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase2/Code/Logs"

    #print(CheckPointPath)
    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(CheckPointPath= CheckPointPath, BasePath= CheckPointPath )
    #print("************************************************")
    #print(SaveCheckPoint)
    #print(ImageSize)
    #print(NumTrainSamples)
    #print(TrainLabels)
    #print(NumClasses)

    


    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    #PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                DivTrain, LatestFile, TrainSet, LogsPath)

    
if __name__ == '__main__':
    main()
 
""" to run the code - python3 Train.py –-CheckPointPath /home/swati/Documents/CV/YourDirectoryID_hw0/Phase2/Code/model —-MiniBatchSize 10

"""