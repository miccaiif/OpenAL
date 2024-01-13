from os import TMP_MAX
import torch
import torch.nn as nn
import numpy as np
from optimizer import optim
from pathlib import Path
# from plot import trainTestPlot
from utils import compute_multiclass_auc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Testing:

    def __init__(self, model, test_dataloader):
        self.model = model
        self.test_dataloader = test_dataloader

    def runner(self):

        self.model.eval()
        with torch.no_grad():

            soft_predict_bank = []

            for index, [images, labels, index] in enumerate(tqdm(self.test_dataloader)):
                images = images.to(device)
                # labels = labels.to(device)
                images = images.view(images.shape[0], 32, -1)
                images = torch.repeat_interleave(images.unsqueeze(dim=1), repeats=3, dim=1)  # batch_size*3*32*12
                outputs = self.model(images)

                softmax_f = nn.Softmax()
                predicted_soft = softmax_f(outputs)

                soft_predict_bank.append(predicted_soft.cpu().detach().numpy())

            index = [b for a in soft_predict_bank for b in a]
            score = np.array(index) # return unlabeled pool score
            # np.save('./soft_predict_bank_0.01_model49.npy', bbb)

        return score


