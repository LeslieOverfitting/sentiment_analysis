from processData import DataProcessor
from transformers import BertTokenizer,BertConfig
from transformers import BertModel
from config import Config
import torch
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from excutor import ModelExcuter
from Model.bert_model_base import BertModel_Base

SEED = 1234

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

    model = BertModel_Base(config.bert_model_path, bert_config, config.max_seq_len).to(config.device)

    modelExcuter = ModelExcuter(train_dataset, dev_dataset, config)
    modelExcuter.train(model)
