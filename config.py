import torch
class Config(object):
    def __init__(self):
        super().__init__()
        self.train_data_path = 'data/train_weibo.csv'
        self.dev_data_path = 'data/dev_weibo.csv'
        self.test_data_path = 'data/test_weibo_clean.csv'
        self.pretrained_path = 'bert_pretrained/'
        #self.bert_model_path = self.pretrained_path + 'bert_base_chinese/bert-base-chinese-pytorch_model.bin'
        #self.bert_config_path = self.pretrained_path + 'bert_base_chinese/bert-base-chinese-config.json'
        #self.bert_vocab_path = self.pretrained_path + 'bert_base_chinese/bert-base-chinese-vocab.txt'
        #self.bert_model_path = 'robert_base_chinese/pytorch_model.bin'
        #self.bert_config_path = 'robert_base_chinese/bert_config.json'
        #self.bert_vocab_path = 'robert_base_chinese/vocab.txt'
        self.bert_model_path = self.pretrained_path + 'ernie/pytorch_model.bin'
        self.bert_config_path = self.pretrained_path + 'ernie/config.json'
        self.bert_vocab_path = self.pretrained_path + 'ernie/vocab.txt'
        self.model_save_path = 'saveModel/Ernie_last3_dropout_AdamW_adv'
        self.max_seq_len = 150
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learn_rate = 1e-5
        self.num_epochs = 3
        self.batch_size = 16
        self.weight_decay = 0.01
        self.adv_type = 'fgm'
        self.max_grad_norm = 1.0