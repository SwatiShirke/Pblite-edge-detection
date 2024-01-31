import os
import torch
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from Network.Network import CIFAR10Model
from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
from Network.Network_2 import DenseNet, Bottleneck
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import sys
import wandb
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def SetupAll():
    # Image Input Shape
    ImageSize = [32, 32, 3]
    return ImageSize

def StandardizeInputs(Img):
    return Img

def ReadImages(Img):
    I1 = Img

    if I1 is None:
        print('ERROR: Image I1 cannot be read')
        sys.exit()

    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1
def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())

    return LabelTest, LabelPred

def Accuracy(Pred, GT):
    """
    Inputs:
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))
def ConfusionMatrix(LabelsTrue, LabelsPred, writer,epoch):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.
    acc = Accuracy(LabelsPred, LabelsTrue)
    # if epoch == 19:
    # Print the confusion matrix as text.
    if epoch == 19:
        for i in range(10):
            print(str(cm[i, :]) + ' ({0})'.format(i))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        # Print the class-numbers for easy reference.
        class_numbers = [" ({0})".format(i) for i in range(10)]
        print("".join(class_numbers))

        print('Accuracy: '+ str(acc), '%')
    writer.add_scalar('TestAccuracy', acc, epoch)

def TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred, criterion, device):
    model = DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)
    model = model.to(device)

    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    print('Number of parameters in this model are %d ' % len(model.state_dict().items()))

    OutSaveT = open(LabelsPathPred, 'w')
    label_set = []
    pred_set = []
    total_loss = 0.0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(TestSet):
            #transform = transforms.ToTensor()
            inputs = inputs.unsqueeze(0).to(device)
            labels = torch.tensor([labels], dtype=torch.long).to(device)  # Convert labels to tensor

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_samples += 1

            _, preds = torch.max(outputs, 1)
            label_set.append(labels.item())
            pred_set.append(preds.item())

            OutSaveT.write(str(preds.item()) + '\n')

    OutSaveT.close()

    avg_loss = total_loss / num_samples
    accuracy = (np.sum(np.array(pred_set) == np.array(label_set)) * 100.0 / len(label_set))

    return label_set, pred_set, avg_loss, accuracy


def main():
    print("abcd")
    ImageSize = SetupAll()
    wandb.init(project="swati_2", entity="svshirke")
    log_path = "./bleh/Densenet"  # Change this to your desired log path
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, 20):
        print('Epoch: ', epoch)
        
        ModelPath = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase2/Code/model/Dense_net.ckpt" 
        LabelsPath = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase2/Code/TxtFiles/LabelsTest.txt"
        LabelsPathPred = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase2/Code/TxtFiles/PredOut.txt"
        TestSet = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=ToTensor())


        label_set, pred_set, loss, accuracy = TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred, criterion, device)
        wandb.log({"LossPerEpoch": loss}, step=epoch)
        wandb.log({"AccPerEpoch": accuracy}, step=epoch)
        writer.add_scalar('Test/Loss', loss, epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)
        LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
        ConfusionMatrix(LabelsTrue, LabelsPred, writer, epoch)
    writer.close()

if __name__ == '__main__':
    main()
