from layers.data_loader import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import TimesNet

warnings.filterwarnings('ignore')

class Exp_Main:
    def __init__(self, args):
        # super(Exp_Classification, self).__init__(args)

        self.args = args
        if self.args.use_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            self.device = torch.device('cpu')
            print('Use CPU')

        ## load data
        self.train_loader = data_provider(self.args, 'train')
        self.val_loader = data_provider(self.args, 'val')
        self.test_loader = data_provider(self.args, 'test')

        ## build model
        self.model = TimesNet.Model(self.args).float().to(self.device)
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.BCELoss()  # nn.CrossEntropyLoss()

    def _get_Acc(self, pred, true):
        pred = torch.Tensor(pred)
        true = torch.Tensor(true)
        prediction = pred >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == true
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        return accuracy

    def vali(self, val_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x)

                pred = outputs.detach().cpu()
                loss = criterion(pred, batch_y)
                total_loss.append(loss)

                preds.append(pred.numpy())
                trues.append(batch_y.numpy())

        total_loss = np.average(total_loss)

        val_pred = np.array(preds)
        val_pred = val_pred.reshape(-1, 1)
        val_y = np.array(trues)
        val_y = val_y.reshape(-1, 1)
        val_Acc = self._get_Acc(val_pred, val_y)
        self.model.train()
        return total_loss, val_Acc

    def train(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        print('train_steps: ', train_steps)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_pred, train_y = [], []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(self.train_loader):
                iter_count += 1
                self.model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                train_loss.append(loss.item())

                train_pred.append(outputs.detach().cpu().numpy())
                train_y.append(batch_y.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                self.model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            val_loss, val_Acc = self.vali(self.val_loader, self.criterion)
            test_loss, test_Acc = self.vali(self.test_loader, self.criterion)

            train_pred = np.array(train_pred)
            train_pred = train_pred.reshape(-1, 1)
            train_y = np.array(train_y)
            train_y = train_y.reshape(-1, 1)
            train_Acc = self._get_Acc(train_pred, train_y)
            print("Epoch: %d, Steps: %d | Train Loss: %.5f Train Acc: %.5f Vali Loss: %.5f Vali Acc: %.5f Test Loss: %.5f Test Acc: %.5f" %
                  (epoch + 1, train_steps, train_loss, train_Acc, val_loss, val_Acc, test_loss, test_Acc))

            early_stopping(-val_Acc, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(self.model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x)

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, 1)
        trues = np.array(trues)
        trues = trues.reshape(-1, 1)
        test_Acc = self._get_Acc(preds, trues)
        print("Test Acc: %.5f" % (test_Acc))
        return
