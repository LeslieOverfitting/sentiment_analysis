from train import train_model
from lstm_model import STModel
import time
import torch
import numpy as np
from config import Config
from process_data import build_data_set, load_word_emb
from dataset_Iterator import DataSetIterator
if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    config = Config()
    train_data, dev_data, test_data = build_data_set(config)
    train_iter = DataSetIterator(config.batch_size, train_data, config.device)
    dev_iter = DataSetIterator(config.batch_size, dev_data, config.device)
    word_emb, word2Index = load_word_emb(config.word_emb_path)
    word_emb = torch.tensor(word_emb, dtype=torch.float,requires_grad=False).to(config.device)
    model = STModel(config.emb_dim, len(word2Index), config.classes_num, config.lstm_hidden_num, config.lstm_layer, config.device, embeddings = word_emb, dropout = config.dropout)
    model.to(config.device)
    print(model.parameters)
    train_model(config, model, train_iter, dev_iter, dev_iter)