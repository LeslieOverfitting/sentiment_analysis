import torch

class Config(object):

    def __init__(self):
        self.train_size_rate = 0.9
        self.seq_length = 140
        self.n_vocab = 0 
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # model 
        self.lstm_layer = 2
        self.lstm_hidden_num = 100
        self.dropout = 0.3
        self.emb_dim = 300
        self.classes_num = 3
        # train
        self.batch_size = 32
        self.epochs = 20
        self.clip = 5
        self.learn_rate = 0.00001
        self.model_save_path = 'STModel.pt'
        # file
        self.word_emb_path = 'data/word_emb'
        self.train_data_clean_path = 'data/train_weibo_clean.csv'
        self.test_data_clean_path = 'data/test_weibo_clean.csv'
        self.train_data_path = 'data/nCoV_100k_train.labled.csv'
        self.test_data_path = 'data/nCoV_10k_test.csv'
        self.preTrain_emb_path = 'data/sgns.weibo.char'
        