import random

import torch
from torch.utils.data import DataLoader
import os
import numpy as np

# Trans
class CRC(torch.utils.data.Dataset):
    def __init__(self, label_flag=None, train=True, gen_cand=False):
        self.train = train
        self.label_flag = label_flag
        self.gen_cand = gen_cand
        # 1. load all features and labels
        save_path = "./embeddings/"
        if train or gen_cand:
            self.all_patches = np.load(os.path.join(save_path, "./training_datasets/embeddings.npy"),allow_pickle=True)
            print(len(self.all_patches))
            self.labels = np.load(os.path.join(save_path, "./training_datasets/labels.npy"),allow_pickle=True)
            self.class2index = np.load(os.path.join(save_path, "./training_datasets/class2index.npy"),allow_pickle=True)
            print('class2index',self.class2index)
            # print('3',len(np.where(self.labels==3)[0]),'6',len(np.where(self.labels==6)[0]),'8',len(np.where(self.labels==8)[0]))
            idx_3 = np.where(self.labels==3)[0]
            idx_3_select = idx_3[np.random.randint(0,len(idx_3),6000)]
            idx_6 = np.where(self.labels == 6)[0]
            idx_6_select = idx_6[np.random.randint(0, len(idx_6), 2000)]
            idx_8 = np.where(self.labels == 8)[0]
            idx_8_select = idx_8[np.random.randint(0, len(idx_8), 10000)]
            idx_imblance = np.where(self.labels==0)[0][:8000]
            idx_imblance = np.append(idx_imblance, (np.where(self.labels==1)[0][:8000]))
            idx_imblance = np.append(idx_imblance, (np.where(self.labels == 2)[0][:8000]))
            idx_imblance = np.append(idx_imblance, idx_3_select)
            idx_imblance = np.append(idx_imblance, (np.where(self.labels == 4)[0][:8000]))
            idx_imblance = np.append(idx_imblance, (np.where(self.labels == 5)[0][:8000]))
            idx_imblance = np.append(idx_imblance, idx_6_select)
            idx_imblance = np.append(idx_imblance, (np.where(self.labels == 7)[0][:8000]))
            idx_imblance = np.append(idx_imblance, idx_8_select)
            print('idx_imblance:',idx_imblance)
            self.patches_imbalence = self.all_patches[idx_imblance]
            self.labels_imbalence = self.labels[idx_imblance]
            print(len(self.patches_imbalence))      # 70288
            self.all_patches = self.patches_imbalence
            self.labels = self.labels_imbalence
            len3=0
            len6=0
            len8=0
            for i in range(len(self.labels)):
                if self.labels[i] == 3:
                    len3+=1
                elif self.labels[i] == 6:
                    len6+=1
                elif self.labels[i] == 8:
                    len8+=1
            print('len',len3, len6, len8)
        else:
            # test pick data from unlabelled pool
            self.all_patches = np.load(os.path.join(save_path, "./testing_datasets/embeddings.npy"),allow_pickle=True)
            self.labels = np.load(os.path.join(save_path, "./testing_datasets/labels.npy"),allow_pickle=True)
            self.class2index = np.load(os.path.join(save_path, "./testing_datasets/class2index.npy"),allow_pickle=True)
            self.all_patches_test = []
            self.labels_test = []
            len3=0
            len6=0
            len8=0
            for i in range(len(self.labels)):
                if self.labels[i] == 3:
                    self.all_patches_test.append(self.all_patches[i])
                    self.labels_test.append(0)
                    len3+=1
                elif self.labels[i] == 6:
                    self.all_patches_test.append(self.all_patches[i])
                    self.labels_test.append(1)
                    len6+=1
                elif self.labels[i] == 8:
                    self.all_patches_test.append(self.all_patches[i])
                    self.labels_test.append(2)
                    len8+=1
            print('len',len3, len6, len8)

        self.num_patches = self.all_patches.shape[0]
        self.num_labels = self.labels.shape[0]
        print("[DATA INFO] labels is {}, num_patches is {}".format(
            self.num_labels, self.num_patches))
        # match with all_patches and their labels
        self.indexes = np.arange(0, self.num_patches, 1)
        self.indexes_tmp = self.indexes.copy()
        self.indexes_unlabelled = []
        if self.train or self.gen_cand:
            if label_flag is None:
                # pick 0.01 percent data to initialize and train the model
                self.label_flag = np.zeros_like(self.labels)
                random.shuffle(self.indexes_tmp)
                # print((int)(len(self.indexes_tmp)*0.01)) # 799
                label_length = int(len(self.indexes_tmp)*0.01)
                for i in range(label_length):
                    self.label_flag[self.indexes_tmp[i]] = 1 # annotated
                # generate unlabelled pool
                for i in range(len(self.indexes)):
                    if self.label_flag[i] == 0:
                        self.indexes_unlabelled.append(i)
            # else:
            #     # label_flag + len(unlabelled_pool)*0.04
            #     for i in range(len(self.indexes)):
            #         if self.label_flag[i] == 0:
            #             self.indexes_unlabelled.append(self.indexes[i]) # collect indexes
            #     random.shuffle(self.indexes_unlabelled)
            #     for i in range((int)(len(self.all_patches) * 0.04)):
            #         if self.label_flag[self.indexes_unlabelled[i]] == 1:
            #             print('error!')
            #         self.label_flag[self.indexes_unlabelled[i]] = 1  # annotated
            #     self.indexes_unlabelled = []
            #     for i in range(len(self.indexes)):
            #         if self.label_flag[i] == 0:
            #             self.indexes_unlabelled.append(i)

            self.class3patches = []
            self.class3labels = []
            self.class3indexes = []
            self.class6indexes = []
            for i in range(len(self.all_patches)):
                if self.label_flag[i] == 1 and (self.labels[i] == 8 or self.labels[i] == 3 or self.labels[i] == 6):
                    self.class3patches.append(self.all_patches[i])
                    self.class3labels.append(self.labels[i])
                    self.class3indexes.append(self.indexes[i])
                if self.label_flag[i] == 1 and self.labels[i] != 3 and self.labels[i] != 6 and self.labels[i] != 8:
                    self.class6indexes.append(self.indexes[i])

            print('first initial model 3class length:',len(self.class3patches))
            # print(self.class3indexes)
            # print(self.class3labels)
            self.unlabelled_patches = []
            self.unlabelled_labels = []
            self.unlabelled_indexes = []
            num = 0
            for i in range(len(self.all_patches)):
                if self.label_flag[i] == 0:
                    self.unlabelled_patches.append(self.all_patches[i])
                    self.unlabelled_labels.append(self.labels[i])
                    self.unlabelled_indexes.append(self.indexes[i])
                else:
                    num += 1
            print("")

    def __getitem__(self, index):
        # patch_image = self.all_patches_pos_slides[index]
        # patch_image = torch.from_numpy(patch_image)
        # each_image_w = torch.nn.functional.dropout(patch_image, p=0.2, training=False)
        # each_image_s = torch.nn.functional.dropout(patch_image, p=0.4, training=False)
        if self.train:
            # pick 3 class which are already annotated(labelled pool)
            patch_feat = self.class3patches[index]
            patch_label = self.class3labels[index]
            patch_index = self.class3indexes[index]
            if patch_label == 3:
                patch_label = 0
            elif patch_label == 6:
                patch_label = 1
            elif patch_label == 8:
                patch_label = 2
            return patch_feat, patch_label, patch_index
        elif self.gen_cand:
            # eval(unlabelled pool)
            patch_feat = self.unlabelled_patches[index]
            patch_label = self.unlabelled_labels[index] # will not be used
            patch_index = self.unlabelled_indexes[index]
            return patch_feat, patch_label, patch_index
        else:
            # test always the same
            patch_feat = self.all_patches_test[index]
            patch_label = self.labels_test[index]  #
            return patch_feat, patch_label

    def __len__(self):
        if self.train:
            return len(self.class3labels)
        elif self.gen_cand:
            return len(self.unlabelled_labels)
        else:
            return len(self.labels_test)

def load_dataset(batch_size=1, label_flag=None):
    # initialize
    train_dataset = CRC(label_flag, train=True)
    first_label_flag = train_dataset.label_flag
    # unlabelled_pool_indexes = train_dataset.unlabelled_indexes
    gen_can_dataset = CRC(label_flag=first_label_flag, train=False, gen_cand=True) # according to ssd to gennerate
    unlabelled_pool_indexes = gen_can_dataset.unlabelled_indexes
    labels = gen_can_dataset.labels
    patches = gen_can_dataset.all_patches
    class3indexes = gen_can_dataset.class3indexes
    class6indexes = gen_can_dataset.class6indexes
    test_dataset = CRC(label_flag=first_label_flag, train=False) # stay the same

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256,
                                               shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    gen_can_dataloader = torch.utils.data.DataLoader(gen_can_dataset, batch_size=256,
                                                   shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                               shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    return train_dataloader, gen_can_dataloader, test_dataloader, unlabelled_pool_indexes, labels, patches, class3indexes, class6indexes, first_label_flag

if __name__ == '__main__':
    train_dataloader, gen_can_dataloader, test_dataloader,_,_,_,_,_,_ = load_dataset()
    for i in train_dataloader:
        print(i)
        break

    for i in test_dataloader:
        print(i)
        break
