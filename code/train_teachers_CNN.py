import torch
from datasets import MnistDataset
import argparse
from models import *
import torch.optim as optim

"""
This script is about a distillation problem. It trains two teachers with a CNN for the MNIST dataset.
Teacher 1 trains a model that goes from 0 to 4.
Teacher 2 trains a model that goes from 5 to 9.
The aim of this script is to make some research from the paper 'Unifying Heterogeneous Classifiers With Distillation'
"""

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=int, default=0.001)
parser.add_argument('--train_file', default="/home/student/noel_aitor/mnist/data/MNIST/processed/training.pt")
parser.add_argument('--save_teachers', default=True)
parser.add_argument('--teachers_path', default="/home/student/noel_aitor/mnist/teachers/")
parser.add_argument('--device', default="cuda")
args = vars(parser.parse_args())
device = args["device"]

# Teacher 1
images, targets = torch.load(args["train_file"])
images = images.view(-1, 1, 28, 28)
trainset0to4 = MnistDataset(images, targets)
idx = trainset0to4.targets <= 4
trainset0to4.images = trainset0to4.images[idx]
trainset0to4.targets = trainset0to4.targets[idx]
trainloader = torch.utils.data.DataLoader(trainset0to4, batch_size=64, shuffle=True, num_workers=2)

teacher0to4 = CNN_Model_v2(num_classes=5, channels=128)
teacher0to4.to(device)

epochs = args["epochs"]
learning_rate = args["lr"]
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(teacher0to4.parameters(), lr=learning_rate, momentum=0.9)

# Run over epochs (1 epoch = visited all items in dataset)
for epoch in range(epochs):

    running_loss = 0.0
    total = 0

    # for i, data in enumerate(trainloader, 0):
    for data in trainloader:
        # Apply the learning rate decay
        if epoch % 100 == 0 and epoch != 0:
            learning_rate = learning_rate * 0.5
            optimizer = optim.SGD(teacher0to4.parameters(),
                                  lr=learning_rate, momentum=0.9)

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = teacher0to4(inputs.float())
        target = labels.to(device).long()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        total += len(data)

        # print statistics
        running_loss += loss.item()
    # print every epoch
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / total))

print('Finished teacher 1 training!')

trainset5to9 = MnistDataset(images, targets)
idx = trainset5to9.targets >= 5
trainset5to9.images = trainset5to9.images[idx]
trainset5to9.targets = trainset5to9.targets[idx]
trainset5to9.targets = trainset5to9.targets - 5
trainloader = torch.utils.data.DataLoader(trainset5to9, batch_size=64, shuffle=True, num_workers=2)

teacher5to9 = CNN_Model_v2(num_classes=5, channels=128)
teacher5to9.to(device)

learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(teacher5to9.parameters(), lr=learning_rate, momentum=0.9)

# Run over epochs (1 epoch = visited all items in dataset)
for epoch in range(epochs):

    running_loss = 0.0
    total = 0

    # for i, data in enumerate(trainloader, 0):
    for data in trainloader:
        # Apply the learning rate decay
        if epoch % 100 == 0 and epoch != 0:
            learning_rate = learning_rate * 0.5
            optimizer = optim.SGD(teacher5to9.parameters(),
                                  lr=learning_rate, momentum=0.9)

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = teacher5to9(inputs.float())
        target = labels.to(device).long()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        total += len(data)

        # print statistics
        running_loss += loss.item()
    # print every epoch
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / total))

print('Finished teacher 2 training!')

if args["save_teachers"]:
    torch.save(teacher0to4, args["teachers_path"] + "teacher0to4_CNN.pt")
    torch.save(teacher5to9, args["teachers_path"] + "teacher5to9_CNN.pt")
    print(f"Teachers stored in {args['teachers_path']}")
