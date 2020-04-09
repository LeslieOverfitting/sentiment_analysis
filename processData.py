import pandas as pd
from tqdm import tqdm
import torch
from dataSet import DataSet

class DataProcessor(object):
    def __init__(self, tokenizer, max_len, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    def get_train_feature(self, data_path):
        train_data_df = self.load_data(data_path)
        input_ids_tensor, attention_mask_tensor, token_type_ids_tensor = self.tokenize_data(train_data_df, '微博中文内容')
        return DataSet(input_ids_tensor, attention_mask_tensor, token_type_ids_tensor, train_data_df['情感倾向'], self.device)
    
    def get_dev_feature(self, data_path):
        dev_data_df = self.load_data(data_path)
        input_ids_tensor, attention_mask_tensor, token_type_ids_tensor = self.tokenize_data(dev_data_df, '微博中文内容')
        return DataSet(input_ids_tensor, attention_mask_tensor, token_type_ids_tensor, dev_data_df['情感倾向'], self.device)

    def get_test_feature(self, data_path):
        test_data_df = self.load_data(data_path, True)
        input_ids_tensor, attention_mask_tensor, token_type_ids_tensor = self.tokenize_data(test_data_df, '微博中文内容')
        return DataSet(input_ids_tensor, attention_mask_tensor, token_type_ids_tensor, None, self.device), test_data_df
    
    def load_data(self, data_path, is_test=False):
        df = pd.read_csv(data_path)
        if is_test:
            columns = ['微博id', '微博中文内容']
        else:
            columns = ['微博中文内容', '情感倾向']
        df = df[columns]
        return df

    def tokenize_data(self, df, column):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for content in tqdm(df[column]):
            inputs = self.tokenizer.encode_plus(text=str(content),
                    add_special_tokens=True,
                    max_length=self.max_len,
                    truncation_strategy="longest_first",
                    return_attention_mask=True,
                    pad_to_max_length=True)
            ids, masks, token_type_id = inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']
            input_ids.append(ids)
            attention_mask.append(masks)
            token_type_ids.append(token_type_id)
        return input_ids, attention_mask, token_type_ids