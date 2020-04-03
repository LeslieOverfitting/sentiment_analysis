import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from sklearn import metrics
import numpy as np
from config import Config 
import pandas as pd 
from utils import get_time_diff

def train_model(config, model, train_iter, test_iter, dev_iter):
    start_time = time.time()
    dev_best_loss = float('inf')
    model.train()
    optimizer = optim.Adam(model.parameters(), lr = config.learn_rate)
    total_batch = 0
    for epoch in range(config.epochs):
        print('epoch [{}/{}]'.format(epoch + 1, config.epochs))
        for i, (trains, labels) in enumerate(train_iter):
            total_batch += 1
            hidden = model.init_hidden(trains.size(0))
            model.zero_grad()
            outputs = model(trains, hidden)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 500 == 0:
                true_label = labels.data.cpu()
                predict = torch.max(outputs, dim=1)[1].cpu().numpy()
                trian_acc = metrics.accuracy_score(true_label, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_best_loss > dev_loss: 
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.model_save_path)
                    improve = '*'
                else:
                    improve = ' '
                time_dif = get_time_diff(start_time)
                msg = 'Iter: {0:>6}, Train loss: {1:>5.3}, Train acc: {2:6.2%}, Dev loss {3: 5.3}, Dev acc: {4: 6.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), trian_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
    test_model(config, model, test_iter)

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            hidden = model.init_hidden(texts.size(0))
            outputs = model(texts, hidden)
            loss = F.cross_entropy(outputs, labels)
            predict = torch.max(outputs, dim=1)[1].cpu().numpy()
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            labels_all = np.append(labels_all,labels)
            predict_all = np.append(predict_all, predict)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc,  loss_total / len(data_iter)


def test_model(config, model, test_iter):
    model.load_state_dict(torch.load(config.model_save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = "Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_diff = get_time_diff(start_time)
    print("Time usage:", time_diff)










