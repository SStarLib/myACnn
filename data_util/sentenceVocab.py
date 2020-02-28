from data_util.vocab import Vocabulary
class SentenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_sent_token="<BEGIN>",
                 end_sent_token="<END>"):
        """
        :param token_to_idx:
        :param unk_token: 未登录词
        :param mask_token: 保持句子定长
        :param begin_sent_token: 句子开头标记
        :param end_sent_token: 句子结尾标记
        """
        super(SentenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        # self._begin_sent_token = begin_sent_token
        # self._end_sent_token = end_sent_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        # self.begin_sent_index = self.add_token(self._begin_sent_token)
        # self.end_sent_index = self.add_token(self._end_sent_token)

    def to_serializable(self):
        contents = super(SentenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'mask_token': self._mask_token,
                         # 'begin_sent_token': self._begin_sent_token,
                         # 'end_sent_token': self._end_sent_token
                         })
        return contents

    def lookup_token(self, token):
        """Retrieve the index associated with the token
            or the UNK index if token isn't present.

        :param token(str): the token to look up
        :return index(int): the index corresponding to the token
        Notes:
            'unk_index' needs to be >=0 (having been added into the Vocabulary)
            for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]
        