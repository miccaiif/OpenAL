import argparse
import torch
import yaml
import pickle
from ResNet import ResNet
from dataset_CRC import load_dataset
from train import Training
from tensorboardX import SummaryWriter
import numpy as np
from scipy.stats import entropy
from test import Testing
from in_and_out_distribution_detector import ssd_select_samples_out_distribution, ssd_select_samples_in_distribution
from generate_candidate import Gen
import os
"""Device Selection"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" Initialize model based on command line argument """
model_parser = argparse.ArgumentParser(description='Image Classification Using PyTorch', usage='[option] model_name')
model_parser.add_argument('--model', type=str, default='resnet')
model_parser.add_argument('--model_save', type=bool, default=True)
model_parser.add_argument('--checkpoint', type=bool, default=True)
model_parser.add_argument('--sam', type=bool, default=False)
model_parser.add_argument('--seed', type=int, default=1)

model_parser.add_argument('--model_pth', type=str, default='./checkpoints/checkpoint0049.pth.tar')

model_parser.add_argument('--summary_name', type=str, default='uncertainty_0.3_3class',
                    help='Name of the tensorboard summmary')
args = model_parser.parse_args()
writer = SummaryWriter(comment=args.summary_name)

"""Loading Config File"""
try:
    stream = open("config.yaml", 'r')
    config = yaml.safe_load(stream)
except FileNotFoundError:
    print("Config file missing")

"""Dataset Initialization"""

# data_initialization = initialize_dataset(image_resolution=config['parameters']['image_resolution'], batch_size=config['parameters']['batch_size'],
#                       MNIST=config['parameters']['MNIST'],train_path=args.train_path,test_path=args.test_path)
# train_dataloader, test_dataloader = data_initialization.load_dataset(transform=True)
#
# first pick intialize
labeled_pool = None
flag = None
all_3_class = None
Acc = {}
Precision = {}
Recall = {}
Pick_list = {}
labels3_num = 0
labels6_num = 0
labels8_num = 0
for epoch in range(7):
    print('al for ',epoch, flag)
    # initialize: label_flag = None
    if epoch == 0:
        train_dataloader, gen_can_dataloader, test_dataloader, unlabelled_pool_indexes, labels, patches, class3indexes, class6indexes, label_flag = load_dataset(label_flag=None)
    else:
        train_dataloader, gen_can_dataloader, test_dataloader, unlabelled_pool_indexes, labels, patches, class3indexes, class6indexes, label_flag = load_dataset(label_flag=flag)
    all_3_class = class3indexes
    # input_channel = next(iter(train_dataloader))[0].shape[1]
    input_channel = 3
    n_classes = config['parameters']['n_classes']
    recall = len(class3indexes) / len(patches)

    precision = len(class3indexes) / (int)(len(patches)*0.01)
    if epoch == 0:
        Precision[0] = precision
        Recall[0] = recall

    """Model Initialization"""
    # input:384 dim features 3*
    model = ResNet(input_channel=input_channel, n_classes=n_classes).to(device) #resnet 18

    print(f'Total Number of Parameters of {args.model.capitalize()} is {round((sum(p.numel() for p in model.parameters()))/1000000, 2)}M')
    # if not args.sam:
    trainer = Training(model=model, optimizer=config['parameters']['optimizer'], learning_rate=config['parameters']['learning_rate'],
                train_dataloader=train_dataloader, num_epochs=config['parameters']['num_epochs'],test_dataloader=test_dataloader,
                model_name=args.model, model_save=args.model_save, checkpoint=args.checkpoint, writer=writer)
    Acc[epoch] = trainer.runner()
    # l = class3indexes.append(class6indexes)
    Pick_list[epoch] = label_flag
    # pick candidate: using pos/neg labeled samples to pick candidate
    # using class3indexes and class6indexes & unlabeledindexes
    embeddings_pos = patches[class3indexes] # (262,384)
    embeddings_neg = patches[class6indexes]
    embeddings_unlabeled = patches[unlabelled_pool_indexes]
    # cal score: return embeddings_unlabeled score
    score1 = ssd_select_samples_in_distribution(embeddings_pos, embeddings_unlabeled)
    score2 = ssd_select_samples_out_distribution(embeddings_neg, embeddings_unlabeled)
    score = score1 - score2
    tindex = np.argsort(score)
    # selected_samples = samples[tindex[:min_num],:]
    cut_length = (int)(0.1 * len(tindex)) #20%
    print(tindex[:cut_length])
    selected_samples_indexes = np.array(unlabelled_pool_indexes)[tindex[:cut_length]] # generate_candidate dataset

    checkpoint = torch.load(args.model_pth)	# 加载模型
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)

    tester = Testing(model=model,test_dataloader=gen_can_dataloader)
    soft_bank = tester.runner()

    soft_bank_uncertainty = entropy(soft_bank,axis=1)

    tindex = np.argsort(-soft_bank_uncertainty) # index
    true_index = np.array(unlabelled_pool_indexes)[tindex]

    pick_list = []
    for i in true_index: # second pick 0.04*79997
        if i in selected_samples_indexes:
            pick_list.append(i)
        if epoch == 0 and len(pick_list) == int(len(patches)*0.04):
            break
        if epoch == 1 and len(pick_list) == int(len(patches)*0.05):
            break
        if epoch == 2 and len(pick_list) == int(len(patches)*0.1):
            break
        if epoch == 3 and len(pick_list) == int(len(patches)*0.1):
            break
        if epoch == 4 and len(pick_list) == int(len(patches)*0.1):
            break
        if epoch == 5 and len(pick_list) == int(len(patches)*0.1):
            break
        if epoch == 6 and len(pick_list) == int(len(patches)*0.1):
            break
    num = 0

    for i in (pick_list):
        if (labels[i] == 3 or labels[i] == 6 or  labels[i] == 8):
            num += 1
            all_3_class.append(i)
        if labels[i] == 3:
            labels3_num += 1
        if labels[i] == 6:
            labels6_num += 1
        if labels[i] == 8:
            labels8_num += 1
    with open(os.path.join('result_all.log'), 'a+') as f:
            f.write('Epoch {}: labels3_num {}, labels6_num {}, labels8_num {}, All num {})\n'.format(epoch
                                                                              , labels3_num, labels6_num, labels8_num, num))

    print('choose:',num)
    # all_3_class.extend(pick_list)
    print('after pick length:',len(all_3_class),len(set(all_3_class)))

    flag = label_flag.copy()
    for i in pick_list:
        if label_flag[i] ==1:
            print('error')
        else:
            flag[i] = 1

    recall = len(all_3_class)/len(patches)

    precision = num/len(pick_list)

    Precision[epoch+1] = precision
    Recall[epoch+1] = recall

# print(pick_list)
print(Acc)
print(Precision)
print(Recall)

with open("rebuttal_ours/" + "known3class" + "_seed" + str(
        args.seed) + "_ours" + ".pkl", 'wb') as f:
    data = {'Acc': Acc, 'Precision': Precision, 'Recall': Recall, 'Picklist': Pick_list}
    pickle.dump(data, f)
# f.close()≠
# after first initialize


# {0: 93.5595667870036, 1: 93.31407942238268, 2: 96.64981949458483, 3: 96.31768953068593, 4: 97.61732851985559, 5: 98.45487364620939, 6: 98.74368231046931}
# {0: 0.34918648310387984, 1: 0.0334375, 2: 0.9505, 3: 0.9990276427281567, 4: 0.7578329989195863, 5: 0.7980109739368999, 6: 0.7648628048780488, 7: 0.4100338696020322}
# {0: 0.010068930672344725, 1: 0.013930491897939297, 2: 0.15114222815691652, 3: 0.41069688548846944, 4: 0.5878956295788372, 5: 0.7558554982135768, 6: 0.900718178209246, 7: 0.9706232631996824}


# {0: 86.4404332129964, 1: 96.51985559566786, 2: 96.98194945848375, 3: 97.2274368231047, 4: 97.84837545126354, 5: 98.07942238267148, 6: 98.58483754512635}
# {0: 0.30183727034120733, 1: 0.6584726319239593, 2: 0.9910854745673833, 3: 0.9909699970870959, 4: 0.9635863408318498, 5: 0.26200323682790866, 6: 0.5422577422577423, 7: 0.22219755826859044}
# {0: 0.0030148909395973154, 1: 0.08080407087949763, 2: 0.21722184127900682, 3: 0.46277382799812333, 4: 0.6776498610559746, 5: 0.7302320545671082, 6: 0.8281785701396658, 7: 0.8643040167454618}
