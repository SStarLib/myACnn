import torch.nn as nn
import torch
class Selector(nn.Module):
    def __init__(self, config):
        super(Selector, self).__init__()
        self.config = config
        self.relation_emb = nn.Embedding(self.config.num_classes, self.config.relation_dim)
        # self.bias = nn.Parameter(torch.Tensor(self.config.num_classes))
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.relation_emb.weight.data)
        # nn.init.normal(self.bias)

    def get_logits(self, x):
        logits = torch.matmul(x, torch.transpose(self.relation_emb.weight, 0, 1), ) + self.bias
        return logits
    def forward(self, relation, all_y):
        """

        :param relation:shape [b_s, 1]; all_y: shape [b_s, num_rel]
        :return: W_L[b_s,1,rel_emb ] W_all_y[b_s,num_rel,rel_emb ]
        """
        W_L = self.relation_emb(relation)
        W_all_y = self.relation_emb(all_y)
        return W_L, self.relation_emb.weight, W_all_y