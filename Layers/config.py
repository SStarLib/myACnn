class Config(object):
    def __init__(self, args, word_vocab_len, rel_vocab_len, max_sen_len, all_y, pre_emb):
        self.word_vocab_len = word_vocab_len  # 词典大小
        self.word_embed_size = args.word_embed_size  # 词向量维度
        self.pos_num = word_vocab_len  # 位置数量
        self.pos_embed_size = args.pos_embed_size  # Pos向量维度
        self.max_sen_len = max_sen_len  # 句子最大长度
        self.win_size = args.win_size  # slide win size
        self.kernel_size = args.kernel_size  # 卷积核大小
        self.hidden_size = args.hidden_size  # d_c
        self.relation_dim = args.relation_dim  # 标签嵌入维度
        self.num_classes = rel_vocab_len  # 关系类别数目
        self.all_y = all_y
        self.pre_emb = pre_emb

    @classmethod
    def setconfig(cls, args, word_vocab_len, rel_vocab_len, max_sen_len, all_y, pre_emb):
        return cls(args,
                   word_vocab_len=word_vocab_len,
                   rel_vocab_len=rel_vocab_len,
                   max_sen_len=max_sen_len,
                   all_y=all_y,
                   pre_emb=pre_emb)
