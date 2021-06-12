import torch
from datasets import MnistDataset
import argparse
from utils import convert_labels


"""
This script is about a distillation problem. It tests two teachers with a simple NN for the MNIST dataset.
Teacher 1 is a trained model that goes from 0 to 4.
Teacher 2 is a trained model that goes from 5 to 9.
The aim of this script is to make some research from the paper 'Unifying Heterogeneous Classifiers With Distillation'
"""


parser = argparse.ArgumentParser()
parser.add_argument('--test_file', default="/home/student/noel_aitor/mnist/data/MNIST/processed/test.pt")
parser.add_argument('--device', default="cuda")
parser.add_argument('--teacher1', default="/home/student/noel_aitor/mnist/teachers/teacher0to4_NN.pt")
parser.add_argument('--teacher2', default="/home/student/noel_aitor/mnist/teachers/teacher5to9_NN.pt")
args = vars(parser.parse_args())
device = args["device"]


# Teacher 1
images, targets = torch.load(args["test_file"])
testset0to4 = MnistDataset(images, targets)
idx = testset0to4.targets <= 4
testset0to4.images = testset0to4.images[idx]
testset0to4.targets = testset0to4.targets[idx]
testloader = torch.utils.data.DataLoader(testset0to4, batch_size=128, shuffle=True, num_workers=2)

teacher0to4 = torch.load(args["teacher1"])
teacher0to4.to(device)

# Run model on test set and determine accuracy
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = torch.flatten(inputs, start_dim=1).to(device)
        target = convert_labels(labels, 5).to(device)
        outputs = teacher0to4(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        _, target = torch.max(target.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()


# Output model accuracy to user
print('Accuracy of teacher 1 on test images: %d %% (%d wrong out of %d)' % (
    100 * correct / total, total - correct, total))


testset5to9 = MnistDataset(images, targets)
idx = testset5to9.targets >= 5
testset5to9.images = testset5to9.images[idx]
testset5to9.targets = testset5to9.targets[idx]
testset5to9.targets = testset5to9.targets - 5
testloader = torch.utils.data.DataLoader(testset5to9, batch_size=128, shuffle=True, num_workers=2)

teacher5to9 = torch.load(args["teacher2"])
teacher5to9.to(device)

# Run model on test set and determine accuracy
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = torch.flatten(inputs, start_dim=1).to(device)
        target = convert_labels(labels, 5).to(device)
        outputs = teacher5to9(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        _, target = torch.max(target.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()


# Output model accuracy to user
print('Accuracy of teacher 2 on test images: %d %% (%d wrong out of %d)' % (
    100 * correct / total, total - correct, total))


