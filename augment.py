# encoding=utf-8
# https://github.com/zhanlaoban/EDA_NLP_for_Chinese
import jieba
import synonyms
import random
from random import shuffle
import pandas as pd
from tqdm import tqdm

#停用词列表，默认使用哈工大停用词表
f = open('stop_words/hit_stopwords.txt')
stop_words = list()
for stop_word in f.readlines():
    stop_words.append(stop_word[:-1])


def synonym_replacement(words, replace_num):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words])) # 非停用词
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms_words = synonyms.nearby(random_word)[0]
        if len(synonyms_words) >= 1:
            synonyms_word = random.choice(synonyms_words[1: min(5, len(synonyms_words))])
            new_words = [synonyms_word if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= replace_num:
            break

    new_sentence = ''.join(new_words)
    return new_sentence

def sentence_augument(sentence, num_aug = 2, alpha_sy = 0.1):
    seg_list = jieba.cut(sentence)
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    num_words = len(words)
    replace_num = max(1, int(num_words *  alpha_sy))
    augmented_sentences = []
    for _ in range(num_aug):
        augmented_sentences.append(synonym_replacement(words, replace_num))
    return augmented_sentences