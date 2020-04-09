from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_diff
import numpy as np
import pandas as pd
from tqdm import tqdm
from optimizedRounder import OptimizedRounder

class ModelExcuter(object):
    def __init__(self, train_dataset, dev_dataset, config):
        super().__init__()
        self.train_tensor_set = TensorDataset(train_dataset.dataset, train_dataset.labels)
        self.dev_tensor_set = TensorDataset(dev_dataset.dataset, dev_dataset.labels)
        self.config = config

    def train(self, model):
        start_time = time.time()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learn_rate)
        total_batch = 0
        criterion = nn.CrossEntropyLoss()
        dev_per_batch = 500
        #dev_best_loss = float('inf')
        f1_score_best = 0
        last_improve = 0
        model.train()
        total_loss = 0
        for epoch in range(self.config.num_epochs):
            print('epoch [{}/{}]'.format(epoch + 1, self.config.num_epochs))
            for (inputs, labels) in DataLoader(self.train_tensor_set, batch_size=self.config.batch_size, shuffle=True):
                total_batch += 1
                model.zero_grad() 
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if total_batch % dev_per_batch == 0:
                    true_labels = labels.data.cpu()
                    predicts = torch.max(outputs.data, dim=1)[1].cpu().numpy()
                    train_acc = metrics.accuracy_score(true_labels, predicts)
                    time_dif = get_time_diff(start_time)
                    dev_acc, dev_loss, report, confusion, f1_score = self.evaluate(model)
                    model.train()
                    if f1_score_best < f1_score:
                        f1_score_best = f1_score
                        improve = '*'
                        torch.save(model.state_dict(), self.config.model_save_path)
                    else:
                        improve = ' '
                    msg = 'Epoch:{0:>2} Iter: {1:>6}, Train Loss: {2:>5.2}, Train Acc: {3:>6.3%},' \
                                ' Dev Loss: {4:>5.2}, Dev Acc: {5:>6.3%}, f1_score: {6:>8.7}, Time: {7} {8}'
                    print(msg.format(epoch + 1, total_batch, loss.item(), train_acc, dev_loss, dev_acc,  f1_score, time_dif, improve))
        self.evaluate_model(model)



    def evaluate(self, model, flag=False):
        if flag:
            model.load_state_dict(torch.load(self.config.model_save_path))
        model.eval()
        labels_all = np.array([], dtype=int)
        predicts_all = np.array([], dtype=int)
        criterion = nn.CrossEntropyLoss()
        loss_total = 0
        with torch.no_grad():
            for (inputs, labels) in DataLoader(dataset=self.dev_tensor_set, batch_size=self.config.batch_size, shuffle=False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss_total += loss.item()
                predicts = torch.max(outputs.data, dim=1)[1].cpu().numpy()
                labels = labels.data.cpu().numpy()
                predicts_all = np.append(predicts_all, predicts)
                labels_all = np.append(labels_all, labels)
        acc = metrics.accuracy_score(labels_all, predicts_all)
        f1_score = metrics.f1_score(labels_all, predicts_all, average='macro')  
        report = metrics.classification_report(labels_all, predicts_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predicts_all)
        return acc, loss_total / len(self.dev_tensor_set) * self.config.batch_size, report, confusion, f1_score

    def predict(self, model, test_dataset, ids, coef=None):
        model.load_state_dict(torch.load(self.config.model_save_path))
        model.eval()
        start_time = time.time()
        if coef is None:
            coef = [1.0, 1.0, 1.0]
        torch_coef = torch.tensor(coef, device=self.config.device).view(-1, 3)
        predicts_all = []
        for inputs,_ in tqdm(DataLoader(dataset=TensorDataset(test_dataset.dataset, test_dataset.labels), batch_size=self.config.batch_size, shuffle=False)):
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            outputs = outputs * torch_coef
            predicts = list(torch.max(outputs.data, dim=1)[1].cpu().numpy() - 1)
            predicts_all = predicts_all + predicts

        time_dif = get_time_diff(start_time)
        print("Time usage:", time_dif)
        result_pd = pd.DataFrame(
            {
                'id': ids,
                'y': predicts_all
            }
        )
        result_pd.to_csv('predict_ans.csv', index=False)
        print("finish !")

    def evaluate_model(self, model):
        start_time = time.time()
        dev_acc, dev_loss, dev_report, dev_confusion, f1_score = self.evaluate(model, True)
        print('未使用指标优化')
        msg = "Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}"
        print(msg.format(dev_loss, dev_acc))
        print("Precision, Recall and F1-Score...")
        print(dev_report)
        print("Confusion Matrix...")
        print(dev_confusion)
        time_diff = get_time_diff(start_time)
        print("Time usage:", time_diff)

    def get_threshold(self, data_set, model):
        model.eval()
        labels_all = np.array([], dtype=int)
        predicts_all = np.array([[1,1,1]], dtype=float)
        criterion = nn.CrossEntropyLoss()
        loss_total = 0
        with torch.no_grad():
            for (inputs, labels) in DataLoader(dataset=data_set, batch_size=self.config.batch_size, shuffle=False):
                outputs = model(inputs)
                outputs = F.softmax(outputs, dim=1) 
                predict_np = outputs.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
                predicts_all = np.append(predicts_all, predict_np, axis=0)
                labels_all = np.append(labels_all, labels)
        predicts_all = predicts_all[1:]
        optimizedRounder = OptimizedRounder()
        optimizedRounder.fit(predicts_all, labels_all)
        coef = optimizedRounder.get_coef()
        return coef
