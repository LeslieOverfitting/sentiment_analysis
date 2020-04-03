from process_data import build_data_set, load_word_emb
from config import Config
from lstm_model import STModel
from train import test_model
from dataset_Iterator import DataSetIterator
import torch


if __name__ == '__main__':
    cofig = Config()
    test_data = build_data_set(config, True)
    test_iter = DataSetIterator(config.batch_size, test_data, config.device)
    word_emb, word2Index = load_word_emb(config.word_emb_path)
    word_emb = torch.tensor(word_emb, dtype=torch.float,requires_grad=False).to(config.device)
    model = STModel(config.emb_dim, len(word2Index), config.classes_num, config.lstm_hidden_num, config.lstm_layer, config.device, embeddings = word_emb, dropout = config.dropout)
    model.to(config.device)
    test_model(config, model, test_iter)