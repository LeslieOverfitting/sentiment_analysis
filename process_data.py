import pandas as pd
import numpy as np
import pickle
import csv
from config import Config


def load_preTrain_emb(file_path):
    """
    output - pre_train_emb dict{'word':list}
    """
    pre_train_emb = {}
    with open(file_path, mode='r', encoding='utf-8') as f:
        word_num, emb_dim = f.readline().split()
        word_num = int(word_num)
        emb_dim = int(emb_dim)
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == emb_dim + 1:
                pre_train_emb[tokens[0]] = list(map(lambda x: float(x), tokens[1:]))
    return pre_train_emb


def build_word_emb(config):
    pre_train_emb = load_preTrain_emb(config.preTrain_emb_path)
    data_train = pd.read_csv(config.train_data_clean_path)
    data_train = data_train[['微博中文内容', '情感倾向']]
    data_test = pd.read_csv(config.test_data_clean_path)
    data_test = data_test[['微博id','微博中文内容']]
    word2Index = {"<PAD>": 0, '<UNK>': 1}
    word_emb = [np.random.uniform(-0.1, 0.1, config.emb_dim) for _ in range(2)]
    build_word2Index(data_train, word2Index, word_emb, pre_train_emb)
    build_word2Index(data_test, word2Index, word_emb, pre_train_emb)
    word_emb = np.asarray(word_emb, dtype='float32')
    save_word_emb(config.word_emb_path, word2Index, word_emb)


def build_data_set(config, is_predict = False):
    word_emb, word2Index = load_word_emb(config.word_emb_path)
    train_data = pd.read_csv(config.train_data_clean_path)
    test_data = pd.read_csv(config.test_data_clean_path)
    def parse_data(parse_data, is_predict = False):
        contents = parse_data['微博中文内容'].apply(str).apply(list)
        contents = contents.apply(lambda text: text[: config.seq_length] + max(0, config.seq_length - len(text)) * ['<UNK>'])
        contents = contents.apply(lambda text: list(map(lambda x: word2Index.get(x, 1), text)))
        if is_predict:
            ids = parse_data['微博id'].apply(lambda x: int(x))
            return list(zip(contents, ids))
        else:
            content_label = parse_data['情感倾向'].apply(lambda x: int(x) + 1)
            return list(zip(contents, content_label))
    if(is_predict):
        test_data = parse_data(test_data, True)
        return test_data
    train_data = parse_data(train_data)
    test_data = parse_data(test_data, True)
    train_data, valid_data = split_train_data(train_data , config.train_size_rate)
    return train_data, valid_data, test_data

    
def split_train_data(data, rate):
    data_size = len(data)
    train_data = data[:int(data_size * rate)]
    valid_data = data[int(data_size * rate):]
    return train_data, valid_data


def save_word_emb(word_emb_path, word2Index, word_emb):
    with open(word_emb_path, 'wb') as f:
        pickle.dump((word_emb, word2Index), f)
    print('save word emb succ')


def load_word_emb(word_emb_path):
    with open(word_emb_path, 'rb') as f:
        word_emb, word2Index = pickle.load(f)
    return word_emb, word2Index


def build_word2Index(data_pd, word2Index, word_emb, pre_train_emb):
    data_pd['微博中文内容'] = data_pd['微博中文内容'].apply(str).apply(list)
    for x in data_pd['微博中文内容']:
        for word in x:
            if word not in word2Index:
                if word in pre_train_emb: # in vocab:
                    word2Index[word] = len(word2Index)
                    word_emb.append(pre_train_emb[word])


def clean_data_set(file_path, is_train = False):
    data = pd.read_csv(file_path, engine ='python')
    if is_train:
        data = data[data['情感倾向'].isin(['0', '1', '-1'])]
    data['微博中文内容'].fillna(' ',inplace= True)
    data.to_csv(file_path, index=False)



if __name__ == '__main__':
    config = Config()
    #clean_data_set(config.train_data_path)
    print('clean data over')
    #build_word_emb(config)
    print('build word emb over')
    train_data, valid_data, test_data = build_data_set(config)
    print('train data size',len(train_data))
    print('valid data size',len(valid_data))
    print('test data size',len(test_data))