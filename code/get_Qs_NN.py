import argparse
import torch.nn as nn
from utils import *
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default="/home/student/noel_aitor/mnist/data/MNIST/processed/training.pt")
parser.add_argument('--t1', default="/home/student/noel_aitor/mnist/teachers/teacher0to4_NN.pt")
parser.add_argument('--t2', default="/home/student/noel_aitor/mnist/teachers/teacher5to9_NN.pt")
args = vars(parser.parse_args())

logging.basicConfig(filename=f"/home/student/noel_aitor/logs/get_Qs_NN.log",
                    format='%(asctime)s - %(message)s', level=logging.INFO)

logging.info("Loading images and teachers...")
images_train, targets_train = torch.load(args["train_file"])
teacher1 = torch.load(args['t1'], map_location="cpu")
teacher2 = torch.load(args['t2'], map_location="cpu")

u_df = pd.DataFrame()
m = nn.Softmax(dim=1)
M = create_MNIST_mask(targets_train)

logging.info("Starting to extract logits...")
for i, img in enumerate(images_train):
    z1 = teacher1(img.reshape(1, 784).float())
    z2 = teacher2(img.reshape(1, 784).float())

    probs_t1 = m(z1).cpu().data.numpy()[0]
    probs_t2 = m(z2).cpu().data.numpy()[0]

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

    if i % 500 == 0:
        logging.info(f"Extracted logits: {i}")

u_df.to_pickle(f"/home/student/noel_aitor/mnist/logits/mnist_qs_NN.csv")
