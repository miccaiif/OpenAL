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
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Testing:

    def __init__(self, model, test_dataloader):
        self.model = model
        self.test_dataloader = test_dataloader

    def runner(self, query_batch):
        queryIndex = []
        labelArr = []
        S_ij = {}

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
                for i in range(images.shape[0]):
                    queryIndex.append(index)
                labelArr += list(np.array(labels.cpu().data))
                v_ij, predicted = outputs.max(1)
                for i in range(len(predicted.data)):
                    tmp_class = np.array(predicted.data.cpu())[i]
                    tmp_index = index[i]
                    tmp_label = np.array(labels.data.cpu())[i]
                    tmp_value = np.array(v_ij.data.cpu())[i]
                    if tmp_class not in S_ij:
                        S_ij[tmp_class] = []
                    S_ij[tmp_class].append([tmp_value, tmp_index, tmp_label])

            tmp_data = []
            for tmp_class in S_ij:
                S_ij[tmp_class] = np.array(S_ij[tmp_class])
                activation_value = S_ij[tmp_class][:, 0]
                if len(activation_value) < 2:
                    continue
                gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
                gmm.fit(np.array(activation_value).reshape(-1, 1))
                prob = gmm.predict_proba(np.array(activation_value).reshape(-1, 1))
                # 得到为known类别的概率
                prob = prob[:, gmm.means_.argmax()]
                # 如果为unknown类别直接为0

                if len(tmp_data) == 0:
                    tmp_data = np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))
                else:
                    tmp_data = np.vstack((tmp_data, np.hstack((prob.reshape(-1, 1), S_ij[tmp_class]))))

            tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]
            tmp_data = tmp_data.T
            queryIndex = tmp_data[2][-query_batch:].astype(int)
            labelArr = tmp_data[3].astype(int)
            queryLabelArr = tmp_data[3][-query_batch:]

        return queryIndex


