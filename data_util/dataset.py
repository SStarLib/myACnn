import json
import torch
from torch.utils.data import Dataset
import pandas as pd
from data_util.vectorizer import SenVectorizer
class SenDataset(Dataset):
    def __init__(self, sen_df, vectorizer, win_size):
        """

        :param sen_df(pandas.DataFrame): the dataset
        :param vectorizer(SenVectorizer): vectorizer instantiated from dataset
        """
        self.sen_df = sen_df
        self._vectorizer = vectorizer
        self._win_size = win_size

        # +1 if only using begin_seq, +2 if using both begin and end seq tokens
        measure_len = lambda context: len(context)
        self._max_seq_length = max(map(measure_len, sen_df.sentence)) #+2

        self.train_df = self.sen_df[self.sen_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.sen_df[self.sen_df.split == 'val']
        self.val_size = len(self.val_df)

        self.test_df = self.sen_df[self.sen_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.val_size),
                             'test': (self.test_df, self.test_size)}
        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, sen_df, win_size):
        """Load dataset and make a new vectorizer from scratch

        :param sen_csv(str): location of the dataset
        :return: an instance of the SenDataset
        """
        # sen_df = pd.read_csv(sen_csv)
        train_news_df = sen_df[sen_df.split=='train']
        return cls(sen_df, SenVectorizer.from_dataframe(train_news_df),win_size)

    @classmethod
    def load_dataset_and_load_vectorizer(cls, sen_data, vectorizer_filepath, win_size):
        """Load dataset and the corresponding vectorizer
        Used in the case in the vectorizer has been cached for re-use

        :param sen_data(pd.DataFrame): location of the dataset
        :param vectorizer_filepath(str): location of the saved vectorizer
        :return: an instance of SenDataset
        """
        # sen_df = pd.read_csv(sen_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(sen_data, vectorizer, win_size)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file

        :param vectorizer_filepath(str): the location of the serialized vectorizer
        :return: an instance of NewsVectorizer
        """
        with open(vectorizer_filepath) as fp:
            return SenVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json

        :param vectorizer_filepath(str): the location to save the vectorizer
        :return:
        """
        with open(vectorizer_filepath, 'w') as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ returns the vectorizer"""
        return self._vectorizer

    def set_split(self, split="train"):
        """selects the splits in the dataset using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        :param index(int): the index to the data point
        :return: a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        epos1 = [int(row.ent1Pos_s), int(row.ent1Pos_e)]
        epos2 = [int(row.ent2Pos_s), int(row.ent2Pos_e)]

        out_vec, e1_vec, e2_vec, d1, d2, e1d2, e2d1, rel_vec   = \
            self._vectorizer.vectorize(ent1Pos=epos1, ent2Pos=epos2,sentence=row.sentence,
                                       relation=row.relation, win_size=self._win_size,
                                       vec_len=self._max_seq_length)

        return {'out_vec':out_vec,'e1_vec':e1_vec, 'e2_vec':e2_vec, 'd1':d1, 'd2':d2,
                'e1d2':e1d2, 'e2d1':e2d1, 'rel_vec':rel_vec}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        :param batch_size(int):
        :return: number of batches in the dataset
        """
        return len(self) // batch_size
