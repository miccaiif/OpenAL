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


class Training:

    def __init__(self, model, optimizer, learning_rate, train_dataloader, num_epochs, writer,
                 test_dataloader, eval=True, plot=True, model_name=None, model_save=False, checkpoint=False):
        self.model = model
        self.learning_rate = learning_rate
        self.optim = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs
        self.eval = eval
        self.plot = plot
        self.model_name = model_name
        self.model_save = model_save
        self.checkpoint = checkpoint
        self.writer = writer

    def runner(self):
        best_accuracy = float('-inf')
        criterion = nn.CrossEntropyLoss()
        if self.model_name in ['alexnet', 'vit', 'mlpmixer', 'resmlp', 'squeezenet', 'senet', 'mobilenetv1', 'gmlp',
                               'efficientnetv2']:
            self.optimizer, scheduler = optim(model_name=self.model_name, model=self.model, lr=self.learning_rate)

        elif self.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        elif self.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        else:
            pass

        train_losses = []
        train_accu = []
        test_losses = []
        test_accu = []
        # Train the model
        total_step = len(self.train_dataloader)

        for epoch in range(self.num_epochs):
            running_loss = 0
            correct = 0
            total = 0

            for i, (images, labels, indexes) in enumerate(tqdm(self.train_dataloader)):
                images = images.to(device) # 256,384
                images = images.view(images.shape[0], 32, -1)
                images = torch.repeat_interleave(images.unsqueeze(dim=1), repeats=3, dim=1) # batch_size*3*32*12
                labels = labels.to(device)
                indexes = indexes.to(device)
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                softmax_f = nn.Softmax()
                predicted_soft = softmax_f(outputs)
                roc_auc = 0

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                train_loss = running_loss / len(self.train_dataloader)
                train_accuracy = 100. * correct / total

                if (i + 1) % 100 == 0:
                    print(
                        'Epoch [{}/{}], Step [{}/{}], Accuracy: {:.3f}, Train Loss: {:.4f}, AUC_Class1: {:.4f}, AUC_Class2: {:.4f}, AUC_Class3: {:.4f}, AUC_Macro: {:.4f}'
                        .format(epoch + 1, self.num_epochs, i + 1, total_step, train_accuracy, loss.item(),
                                roc_auc[0], roc_auc[1], roc_auc[2], roc_auc["macro"]))

                self.writer.add_scalar('Train/Loss', loss.item(), i + 1)
                self.writer.add_scalar('Train/Accuracy', train_accuracy, i + 1)
                # self.writer.add_scalar('Train/AUC_Class1', roc_auc[0], i + 1)
                # self.writer.add_scalar('Train/AUC_Class2', roc_auc[1], i + 1)
                # self.writer.add_scalar('Train/AUC_Class3', roc_auc[2], i + 1)
                # self.writer.add_scalar('Train/AUC_Macro', roc_auc["macro"], i + 1)

            if self.eval:
                self.model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    running_loss = 0
                    predicted_soft_all = []
                    labels_all = []

                    for images, labels in tqdm(self.test_dataloader):
                        images = images.to(device)
                        labels = labels.to(device)
                        images = images.view(images.shape[0], 32, -1)
                        images = torch.repeat_interleave(images.unsqueeze(dim=1), repeats=3, dim=1)  # batch_size*3*32*12
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        running_loss += loss.item()

                        softmax_f = nn.Softmax()
                        predicted_soft = softmax_f(outputs)

                        predicted_soft_all.append(predicted_soft.cpu().detach().numpy())
                        labels_all.append(labels.cpu().detach().numpy())

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        test_loss = running_loss / len(self.test_dataloader)
                        test_accuracy = (correct * 100) / total

                    predicted_soft_all_ = [b for a in predicted_soft_all for b in a]
                    predicted_soft_all_ = np.array(predicted_soft_all_)

                    labels_all_ = [b for a in labels_all for b in a]
                    labels_all_ = np.array(labels_all_)

                    roc_auc = compute_multiclass_auc(predicted_soft_all_,labels_all_, n_classes=3)

                    print(
                        'Epoch: %.0f | Test Loss: %.3f | Accuracy: %.3f | AUC_Class1: %.3f | AUC_Class2: %.3f | AUC_Class3: %.3f | AUC_Macro: %.3f' % (
                        epoch + 1, test_loss, test_accuracy, roc_auc[0], roc_auc[1], roc_auc[2],
                        roc_auc["macro"]))

                    self.writer.add_scalar('Test/Loss', test_loss, epoch + 1)
                    self.writer.add_scalar('Test/Accuracy', test_accuracy, epoch + 1)
                    self.writer.add_scalar('Test/AUC_Class1', roc_auc[0], epoch + 1)
                    self.writer.add_scalar('Test/AUC_Class2', roc_auc[1], epoch + 1)
                    self.writer.add_scalar('Test/AUC_Class3', roc_auc[2], epoch + 1)
                    self.writer.add_scalar('Test/AUC_Macro', roc_auc["macro"], epoch + 1)

            if test_accuracy > best_accuracy and self.model_save:
                best_accuracy = test_accuracy # lack
                Path('model_store/').mkdir(parents=True, exist_ok=True)
                # torch.save(self.model, 'model_store/'+self.model_name+'_best-model.pt')
                torch.save(self.model.state_dict(), 'model_store/' + self.model_name + 'best-model-parameters.pt')

            for p in self.optimizer.param_groups:
                print(f"Epoch {epoch + 1} Learning Rate: {p['lr']}")

            if self.checkpoint:
                path = 'checkpoints/checkpoint{:04d}.pth.tar'.format(epoch)
                Path('checkpoints/').mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        'epoch': self.num_epochs,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss
                    }, path
                )

            train_accu.append(train_accuracy)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accu.append(test_accuracy)
        return best_accuracy
        # trainTestPlot(self.plot, train_accu, test_accu, train_losses, test_losses, self.model_name)
