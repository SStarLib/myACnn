import os

from argparse import Namespace
import collections
import nltk.data
import numpy as np
import pandas as pd
import re
import string
from tqdm import tqdm
from pre_util import load_data, clean_text, load_predata


def preprocess_data():
    args = Namespace(
        train_dataset="data/dataset/train.txt",
        train_pre_dataset="data/dataset_pre/train.txt",
        test_pre_dataset="data/dataset_pre/test.txt",
        window_size=3,
        train_proportion=0.7,
        val_proportion=0.15,
        test_proportion=0.15,
        output_munged_csv="data/dataset/sen_with_pos_splits.csv",
        seed=1337
    )
    # Split the raw text book into sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    relations = load_data(args.train_dataset, tokenizer)
    sentences, e1_pos_s, e1_pos_e, e2_pos_s, e2_pos_e = load_predata(args.train_pre_dataset, tokenizer)
    # cleaned_sentences = [clean_text(token) for sentence in context for token in sentence]
    data = []
    for sentence, relation, e1_p_s, e1_p_e, e2_p_s, e2_p_e in zip(sentences, relations, e1_pos_s, e1_pos_e, e2_pos_s,
                                                                  e2_pos_e):
        data.append([e1_p_s, e1_p_e, e2_p_s, e2_p_e, sentence, relation])
    sen_data = pd.DataFrame(data, columns=["ent1Pos_s", "ent1Pos_e", "ent2Pos_s", "ent2Pos_e", "sentence", "relation"])

    # Create split data
    n = len(sen_data)

    def get_split(row_num):
        if row_num <= n * args.train_proportion:
            return 'train'
        elif (row_num > n * args.train_proportion) and (row_num <= n * args.train_proportion + n * args.val_proportion):
            return 'val'
        else:
            return 'test'


    sen_data['split'] = sen_data.apply(lambda row: get_split(row.name), axis=1)
    sen_data.head()

    # sen_data.to_csv(args.output_munged_csv, index=False, quoting=0)
    return sen_data

if __name__ == '__main__':
    preprocess_data()