import torch

class Config(object):
    def __init__(self):
        super().__init__()
        self.train_data_path = 'data/train_weibo.csv'
        self.dev_data_path = 'data/dev_weibo.csv'
        self.test_data_path = 'data/test_weibo.csv'
        self.bert_model_path = 'bert_base_chinese/bert-base-chinese-pytorch_model.bin'
        self.bert_config_path = 'bert_base_chinese/bert-base-chinese-config.json'
        self.bert_vocab_path = 'bert_base_chinese/bert-base-chinese-vocab.txt'
        self.model_save_path = 'SaveModel/bert_model'
        self.max_seq_len = 140
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learn_rate = 1e-5
        self.num_epochs = 3
        self.batch_size = 16