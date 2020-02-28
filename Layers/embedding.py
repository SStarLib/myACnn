import torch.nn as nn
import torch
class Embedding(nn.Module):
    def __init__(self, embed_config):
        super(Embedding, self).__init__()
        self.config = embed_config
        self.word_embed = nn.Embedding(num_embeddings=self.config.word_vocab_len,
                                           embedding_dim=self.config.word_embed_size)
        self.pos1_embed = nn.Embedding(num_embeddings=self.config.max_sen_len*2,
                                       embedding_dim=self.config.pos_embed_size)
        self.pos2_embed = nn.Embedding(num_embeddings=self.config.pos_num,
                                       embedding_dim=self.config.pos_embed_size)
        self.init_word_weights()
        self.init_pos_weights()

    def init_word_weights(self):
        self.word_embed.weight.data.copy_(torch.from_numpy(self.config.pre_emb))

    def init_pos_weights(self):
        nn.init.xavier_uniform(self.pos1_embed.weight.data)
        if self.pos1_embed.padding_idx is not None:
            self.pos1_embed.weight.data[self.pos1_embed.padding_idx].fill_(0)
        nn.init.xavier_uniform(self.pos2_embed.weight.data)
        if self.pos2_embed.padding_idx is not None:
            self.pos2_embed.weight.data[self.pos2_embed.padding_idx].fill_(0)
    def forward(self, word, pos1, pos2, entity1, entity2):
        word =self.word_embed(word)
        pos1 = self.pos1_embed(pos1)
        pos2 = self.pos2_embed(pos2)
        entity1 = self.word_embed(entity1)
        entity2 = self.word_embed(entity2)
        embedding = torch.cat((word, pos1, pos2), dim=2)
        return embedding, word, entity1, entity2