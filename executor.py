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
from transformers import get_linear_schedule_with_warmup, AdamW
from model.adversal import FGM

class ModelExcuter(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def train(self, model, train_loader, dev_loader, fold_num, use_weight=False):
        start_time = time.time()
        t_total = len(train_loader) * self.config.num_epochs
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        warmup_steps = int(t_total * 0.1)
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learn_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
        total_batch = 0
        if use_weight:
            weight= torch.tensor([2, 1, 1.5], dtype=torch.float).to(self.config.device)
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            criterion = nn.CrossEntropyLoss()
        dev_per_batch = 500
        #dev_best_loss = float('inf')
        f1_score_best = 0
        last_improve = 0
        model.train()
        total_loss = 0
        if self.config.adv_type == 'fgm':
            fgm = FGM(model)
        for epoch in range(self.config.num_epochs):
            print('epoch [{}/{}]'.format(epoch + 1, self.config.num_epochs))
            for data_batch in train_loader:
                # 转化为 tensor
                input_ids = data_batch[0].clone().detach().to(self.config.device)
                attention_masks = data_batch[1].clone().detach().to(self.config.device)
                token_type_ids = data_batch[2].clone().detach().to(self.config.device)
                labels = data_batch[3].clone().detach().long().to(self.config.device)
                model_inputs = (input_ids, attention_masks, token_type_ids)
                total_batch += 1
                model.zero_grad() 
                outputs = model(model_inputs)
                loss = criterion(outputs, labels.view(-1))
                loss.backward()
                # 对抗
                if self.config.adv_type == 'fgm':
                    fgm.attack()  ##对抗训练
                    adv_outputs = model(model_inputs)
                    loss_adv = criterion(adv_outputs, labels.view(-1))
                    loss_adv.backward()
                    fgm.restore()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                if total_batch % dev_per_batch == 0:
                    true_labels = labels.data.cpu()
                    predicts = torch.max(outputs.data, dim=1)[1].cpu().numpy()
                    train_acc = metrics.accuracy_score(true_labels, predicts)
                    time_dif = get_time_diff(start_time)
                    dev_acc, dev_loss, report, confusion, f1_score = self.evaluate(model, dev_loader, fold_num)
                    model.train()
                    if f1_score_best < f1_score:
                        f1_score_best = f1_score
                        improve = '*'
                        torch.save(model.state_dict(), self.config.model_save_path + "-fold" + str(fold_num))
                        torch.save(model.state_dict(), 'saveModel/temp')
                    else:
                        improve = ' '
                        if f1_score_best - 0.02 > f1_score:
                            improve = '-'
                            model.load_state_dict(torch.load('saveModel/temp'))
                        else:
                            torch.save(model.state_dict(), 'saveModel/temp')
                    msg = 'Epoch:{0:>2} Iter: {1:>6}, Train Loss: {2:>5.2}, Train Acc: {3:>6.3%},' \
                                ' Dev Loss: {4:>5.2}, Dev Acc: {5:>6.3%}, f1_score: {6:>8.7}, Time: {7} {8}'
                    print(msg.format(epoch + 1, total_batch, loss.item(), train_acc, dev_loss, dev_acc,  f1_score, time_dif, improve))
        self.evaluate_model(model, dev_loader, fold_num)



    def evaluate(self, model, dev_loader, fold_num, flag=False, use_weight=False):
        if flag:
            model.load_state_dict(torch.load(self.config.model_save_path + "-fold" + str(fold_num)))
        model.eval()
        labels_all = np.array([], dtype=int)
        predicts_all = np.array([], dtype=int)
        if use_weight:
            weight= torch.tensor([2, 1, 1.5], dtype=torch.float).to(self.config.device)
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            criterion = nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss()
        loss_total = 0
        with torch.no_grad():
            for data_batch in dev_loader:
                input_ids = data_batch[0].clone().detach().to(self.config.device)
                attention_masks = data_batch[1].clone().detach().to(self.config.device)
                token_type_ids = data_batch[2].clone().detach().to(self.config.device)
                labels = data_batch[3].clone().detach().long().to(self.config.device)
                model_inputs = (input_ids, attention_masks, token_type_ids)
                outputs = model(model_inputs)
                loss = criterion(outputs, labels.view(-1))
                loss_total += loss.item()
                predicts = torch.max(outputs.data, dim=1)[1].cpu().numpy()
                labels = labels.data.cpu().numpy()
                predicts_all = np.append(predicts_all, predicts)
                labels_all = np.append(labels_all, labels)
        acc = metrics.accuracy_score(labels_all, predicts_all)
        f1_score = metrics.f1_score(labels_all, predicts_all, average='macro')  
        report = metrics.classification_report(labels_all, predicts_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predicts_all)
        return acc, loss_total / len(dev_loader), report, confusion, f1_score

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

    def predict_k_fold(self, model, test_loader, ids, fold_index, coef=None):
        model.load_state_dict(torch.load(self.config.model_save_path + "-fold" + str(fold_index)))
        model.eval()
        start_time = time.time()
        if coef is None:
            coef = [1.0, 1.0, 1.0]
        torch_coef = torch.tensor(coef, device=self.config.device).view(-1, 3)
        predicts_all = np.random.randn(1,3)
        with torch.no_grad():
            for data_batch in tqdm(test_loader):
                input_ids = data_batch[0].clone().detach().to(self.config.device)
                attention_masks = data_batch[1].clone().detach().to(self.config.device)
                token_type_ids = data_batch[2].clone().detach().to(self.config.device)
                model_inputs = (input_ids, attention_masks, token_type_ids)
                outputs = model(model_inputs)
                outputs = F.softmax(outputs, dim=1)
                outputs = outputs * torch_coef
                predicts = outputs.cpu().numpy()
                predicts_all = np.concatenate((predicts_all, predicts), axis=0)
        
        time_dif = get_time_diff(start_time)
        print("Time usage:", time_dif)
        predicts_all = np.delete(predicts_all, 0, axis=0)
        result_pd = pd.DataFrame(
            {
                '-1':predicts_all.T[0],
                '0':predicts_all.T[1],
                '1':predicts_all.T[2],
            },
            index=ids
        )
        save_name = self.config.predict_save_path + "-_fold" + str(fold_index) + ".csv"
        result_pd.to_csv(save_name)

    def evaluate_model(self, model, dev_loader, fold_num):
        start_time = time.time()
        dev_acc, dev_loss, dev_report, dev_confusion, f1_score = self.evaluate(model, dev_loader, fold_num, flag=True)
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
