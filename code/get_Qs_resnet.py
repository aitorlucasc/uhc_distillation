import torch.nn as nn
import torch
import pandas as pd
from datasets import MnistDataset
from utils import *
import argparse
import logging



parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default="/home/student/noel_aitor/mnist/data/MNIST/processed/training.pt")
parser.add_argument('--t1', default="/home/student/noel_aitor/mnist/teachers/teacher0to4_resnet.pt")
parser.add_argument('--t2', default="/home/student/noel_aitor/mnist/teachers/teacher5to9_resnet.pt")
args = vars(parser.parse_args())

logging.basicConfig(filename=f"/home/student/noel_aitor/logs/get_Qs_resnet.log",
                    format='%(asctime)s - %(message)s', level=logging.INFO)


images_train, targets_train = torch.load(args["train_file"])
trainset = MnistDataset(images_train, targets_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)

teacher1 = torch.load(args['t1'], map_location="cpu")
teacher2 = torch.load(args['t2'], map_location="cpu")

u_df = pd.DataFrame()
m = nn.Softmax(dim=0)
M = create_MNIST_mask(targets_train)

counter = 0
logging.info("Starting to extract logits...")

for images, labels in trainloader:
    z1_batch = teacher1(images.view(-1, 1, 28, 28).float())
    z2_batch = teacher2(images.view(-1, 1, 28, 28).float())
    
    for z1, z2 in zip(z1_batch, z2_batch):
        probs_t1 = m(z1).cpu().data.numpy()
        probs_t2 = m(z2).cpu().data.numpy()

        P, Z = get_PZ_matrices_csv(M, z1, z2, probs_t1, probs_t2)


        # Get u from Cross-Entropy method 1
        u_CE = ce_method1_csv(probs_t1, probs_t2)

        # Get u from MF probability space method 2
        u_MFPS, _ = mf_prob_space(M, P)

        # Get u from MF logit space method 2
        u_MFLS, _, _ = mf_logit_space(M, Z)

        u_df = u_df.append({
            "u_CE": np.array(u_CE),
            "u_MFPS": np.array(u_MFPS),
            "u_MFLS": np.array(u_MFLS)
        }, ignore_index=True)

    logging.info(f"Extracted logits: {counter + 64}/60000")
    counter += 64

        
u_df.to_pickle(f"/home/student/noel_aitor/mnist/logits/mnist_qs_resnet.csv")
