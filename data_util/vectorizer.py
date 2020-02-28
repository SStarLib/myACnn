import re

import numpy as np
from collections import Counter
from data_util.vocab import Vocabulary
from data_util.sentenceVocab import SentenceVocabulary
import string
from general_util import pos, createWin
import nltk


class SenVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use """

    def __init__(self, sentence_vocab, relation_vocab):
        self.sent_vocab = sentence_vocab
        self.rel_vocab = relation_vocab

    def vectorize(self, ent1Pos, ent2Pos, sentence, relation, win_size, vec_len=-1):
        """
        :param sentence(str): the string of words separated by a space
        :param vector_length(int): an argument for forcing the length of index vector
        :return: the vectorized sentence (numpy.array), the vectorized relation(numpy.array)
        """
        d1, d2, e1d2, e2d1 = [], [], [], []
        indices = [self.sent_vocab.lookup_token(token)
                   for token in sentence]
        len_idx = len(indices)
        e1_idxs = indices[ent1Pos[0] : ent1Pos[1] + 1]
        e2_idxs = indices[ent2Pos[0] : ent2Pos[1] + 1]

        d1.extend(pos(i - ent1Pos[0]) for i in range(ent1Pos[0]))
        d1.extend(61 for i in range(ent1Pos[0]-1, ent1Pos[1]))
        d1.extend(pos(i - ent1Pos[1]) for i in range(ent1Pos[1] + 1,len_idx))

        d2.extend(pos(i - ent2Pos[0]) for i in range(ent2Pos[0]))
        d2.extend(61 for i in range(ent2Pos[0], ent2Pos[1] + 1))
        d2.extend(pos(i - ent2Pos[1]) for i in range(ent2Pos[1] + 1, len_idx))

        if vec_len <0:
            vec_len = len(indices)
        out_vec = np.zeros(vec_len, dtype=np.int64)
        d1_vec = np.zeros(vec_len, dtype=np.int64)
        d2_vec = np.zeros(vec_len, dtype=np.int64)

        out_vec[:len(indices)]=indices
        out_vec[len(indices):]=self.sent_vocab.mask_index
        d1_vec[:len(indices)]=d1
        d2_vec[:len(indices)]=d2

        # 此处加窗操作，对实现模型会有阻碍
        # mask_list = [self.sent_vocab.mask_index for _ in range(win_size)]
        # out_vec = np.zeros((vec_len-win_size+1, win_size), dtype=np.int64)
        # d1_vec = np.zeros((vec_len-win_size+1, win_size), dtype=np.int64)
        # d2_vec = np.zeros((vec_len-win_size+1, win_size), dtype=np.int64)
        # out_vec[:len_idx-2] = createWin(indices, win_size)
        # out_vec[len_idx-2:] = mask_list
        # d1_vec[:len_idx-2] = createWin(d1, win_size)
        # d1_vec[len_idx-2:] = mask_list
        # d2_vec[:len_idx-2] = createWin(d2, win_size)
        # d2_vec[len_idx-2:] = mask_list

        e1d2.append(pos(ent1Pos[1] - ent2Pos[1]))
        e2d1.append(pos(ent2Pos[1] - ent1Pos[1]))

        e1_vec, e2_vec = np.zeros(5, dtype=np.int64),np.zeros(5, dtype=np.int64)
        e1_len, e2_len = len(e1_idxs), len(e2_idxs)
        e1_vec[:e1_len] = e1_idxs
        e2_vec[:e2_len] = e2_idxs
        e1_vec[e1_len:] = self.sent_vocab.mask_index
        e1_vec[e2_len:] = self.sent_vocab.mask_index

        e1d2 = np.array(e1d2, dtype=np.int64)
        e2d1 = np.array(e2d1, dtype=np.int64)

        # one_hot_matrix_size = (len(self.rel_vocab))
        # one_hot_matrix = np.zeros(one_hot_matrix_size, dtype=np.int64)
        # relation_index = self.rel_vocab.lookup_token(relation)
        # one_hot_matrix[relation_index] = 1
        one_hot_matrix = np.array([self.rel_vocab.lookup_token(relation)], dtype=np.int64)

        return out_vec, e1_vec, e2_vec, d1_vec, d2_vec, e1d2, e2d1, one_hot_matrix

    @classmethod
    def from_dataframe(cls, sen_df, cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe

        :param sen_df (pandas.DataFrame): the target dataset
        :param cutoff (int): frequency threshold for including in Vocabulary
        :return: an instance of the NewsVectorizer
        """
        relation_vocab = Vocabulary()
        mask_token = ""
        for rel in sorted(set(sen_df.relation)):
            relation_vocab.add_token(rel)

        word_counts = Counter()
        for i, sent in enumerate(sen_df.sentence):
            for token in sent:
                if token not in string.punctuation:
                    word_counts[token] += 1

        sentence_vocab = SentenceVocabulary()
        for word, word_count in word_counts.items():
            # 去除低频词
            # if word_count > cutoff:
            #     sentence_vocab.add_token(word)
            sentence_vocab.add_token(word)

        return cls(sentence_vocab, relation_vocab)

    @classmethod
    def from_serializable(cls, contents):
        sentence_vocab = \
            SentenceVocabulary.from_serializable(contents['sentence_vocab'])
        relation_vocab = \
            SentenceVocabulary.from_serializable(contents['relation_vocab'])

        return cls(sentence_vocab=sentence_vocab, relation_vocab=relation_vocab)

    def to_serializable(self):
        return {'sentence_vocab': self.sent_vocab.to_serializable(),
                'relation_vocab': self.rel_vocab.to_serializable()}


