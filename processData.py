import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
class DataProcessor(object):
    def __init__(self, tokenizer, max_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def get_train_feature(self, data_path):
        train_data_df = self.load_data(data_path)
        input_ids_tensor, attention_mask_tensor, token_type_ids_tensor = self.tokenize_data(train_data_df, '微博中文内容')
        return (input_ids_tensor, attention_mask_tensor, token_type_ids_tensor), np.asarray(train_data_df['情感倾向'].astype(int) + 1)
    
    def get_dev_feature(self, data_path):
        dev_data_df = self.load_data(data_path)
        input_ids_tensor, attention_mask_tensor, token_type_ids_tensor = self.tokenize_data(dev_data_df, '微博中文内容')
        return (input_ids_tensor, attention_mask_tensor, token_type_ids_tensor),  np.asarray(dev_data_df['情感倾向'].astype(int) + 1)

    def get_test_feature(self, data_path):
        test_data_df = self.load_data(data_path, True)
        input_ids_tensor, attention_mask_tensor, token_type_ids_tensor = self.tokenize_data(test_data_df, '微博中文内容')
        return (input_ids_tensor, attention_mask_tensor, token_type_ids_tensor), test_data_df
    
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
                    )
            ids =  inputs["input_ids"]
            masks = inputs["attention_mask"]
            segments = inputs["token_type_ids"]
            padding_length = self.max_len - len(ids)
            # 填充
            padding_id = self.tokenizer.pad_token_id
            ids = ids + ([padding_id] * padding_length)
            masks = masks +  ([0] * padding_length)
            segments = segments + ([0] * padding_length)
            input_ids.append(ids)
            attention_mask.append(masks)
            token_type_ids.append(segments)
        return np.asarray(input_ids), np.asarray(attention_mask), np.asarray(token_type_ids)