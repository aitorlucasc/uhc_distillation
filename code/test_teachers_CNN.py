import torch
from datasets import MnistDataset
import argparse
import torch.nn.functional as F


"""
This script is about a distillation problem. It tests two teachers with a CNN for the MNIST dataset.
Teacher 1 is a trained model that goes from 0 to 4.
Teacher 2 is a trained model that goes from 5 to 9.
The aim of this script is to make some research from the paper 'Unifying Heterogeneous Classifiers With Distillation'
"""


parser = argparse.ArgumentParser()
parser.add_argument('--test_file', default="/home/student/noel_aitor/mnist/data/MNIST/processed/test.pt")
parser.add_argument('--device', default="cuda")
parser.add_argument('--teacher1', default="/home/student/noel_aitor/mnist/teachers/teacher0to4_CNN.pt")
parser.add_argument('--teacher2', default="/home/student/noel_aitor/mnist/teachers/teacher5to9_CNN.pt")
args = vars(parser.parse_args())
device = args["device"]


# Teacher 1
images, targets = torch.load(args["test_file"])
images = images.view(-1, 1, 28, 28)

testset0to4 = MnistDataset(images, targets)
idx = testset0to4.targets <= 4
testset0to4.images = testset0to4.images[idx]
testset0to4.targets = testset0to4.targets[idx]
testloader = torch.utils.data.DataLoader(testset0to4, batch_size=64, shuffle=True, num_workers=2)

teacher0to4 = torch.load(args["teacher1"])
teacher0to4.to(device)

# Run model on test set and determine accuracy
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = teacher0to4(data.float())
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(testloader.dataset)

print('Teacher 1: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, len(testloader.dataset),
    100. * correct / len(testloader.dataset)))


testset5to9 = MnistDataset(images, targets)
idx = testset5to9.targets >= 5
testset5to9.images = testset5to9.images[idx]
testset5to9.targets = testset5to9.targets[idx]
testset5to9.targets = testset5to9.targets - 5
testloader = torch.utils.data.DataLoader(testset5to9, batch_size=64, shuffle=True, num_workers=2)

teacher5to9 = torch.load(args["teacher2"])
teacher5to9.to(device)

# Run model on test set and determine accuracy
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = teacher5to9(data.float())
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(testloader.dataset)

print('Teacher 2: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, len(testloader.dataset),
    100. * correct / len(testloader.dataset)))
