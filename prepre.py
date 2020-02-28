
import os
from argparse import Namespace
import collections
import nltk.data
import numpy as np
import pandas as pd
import re
import string
from tqdm import tqdm
from pre_util import load_data,clean_text,load_predata

args = Namespace(
    train_pre_dataset="data/dataset_pre/train.txt",
    test_pre_dataset="data/dataset_pre/test.txt",
    window_size=3,
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="data/dataset/sen_with_splits.csv",
    seed=1337
)
# Split the raw text book into sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences, relations, e1_pos, e2_pos= load_predata(args.train_pre_dataset, tokenizer)

data = [[sent, rel, e1_p, e2_p]
        for sent, rel, e1_p, e2_p in
        zip(sentences, relations, e1_pos, e2_pos)]
sen_data = pd.DataFrame(data, columns=["sentence", "relation", "e1_pos", "e2_pos"])
# Create split data
n = len(sen_data)
def get_split(row_num):
    if row_num <= n*args.train_proportion:
        return 'train'
    elif (row_num > n*args.train_proportion) and (row_num <= n*args.train_proportion + n*args.val_proportion):
        return 'val'
    else:
        return 'test'
sen_data['split']= sen_data.apply(lambda row: get_split(row.name), axis=1)
sen_data.head()
sen_data.to_csv(args.output_munged_csv, index=False)
