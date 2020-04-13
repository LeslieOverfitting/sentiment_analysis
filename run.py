from processData import DataProcessor
from transformers import BertTokenizer,BertConfig
from transformers import BertModel
from config import Config
import torch
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from executor import ModelExcuter
from model.bert_model_base import BertModel_Base
from model.robert_model_base import RoBertModel_Base
from model.ernie_model_base import ErnieModel_Base
from model.ernie_model_pool_last3 import Ernie_PoolLast3_Model
from model.ernie_poollast3_multidp import Ernie_poollast3_multidp

SEED = 12345

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    config = Config()
    bert_config = BertConfig.from_pretrained(config.bert_config_path, output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained(config.bert_vocab_path)
    dataProcessor = DataProcessor(tokenizer, config.max_seq_len, config.device)
    train_dataset = dataProcessor.get_train_feature(config.train_data_path)
    dev_dataset = dataProcessor.get_dev_feature(config.dev_data_path)
    test_dataset, test_df = dataProcessor.get_test_feature(config.test_data_path)
    model = Ernie_poollast3_multidp(config.bert_model_path, bert_config, config.max_seq_len).to(config.device)
    modelExcuter = ModelExcuter(train_dataset, dev_dataset, config)
    modelExcuter.train(model, use_weight=False)
    modelExcuter.predict(model, test_dataset, test_df['微博id'])


