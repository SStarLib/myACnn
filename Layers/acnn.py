import torch.nn as nn
import os
from Layers.embedding import Embedding
from Layers.encoder import Slide, ConvNet
from Layers.atten_input import Atten_input
from Layers.attn_pool import Attn_pool
from Layers.selector import Selector
import torch
import numpy as np
import torch.nn.functional as F
class ACNN(nn.Module):
    def __init__(self,config):
        super(ACNN, self).__init__()
        self.config = config
        self.emb = Embedding(self.config)
        self.slide = Slide(self.config)
        self.atten_input = Atten_input(self.config)
        self.conv = ConvNet(self.config)
        self.selector = Selector(self.config)
        self.atten_pool = Attn_pool(self.config)
        self.rel_emb_weight = None
        self.all_y = torch.from_numpy(np.array(self.config.all_y))
    def predict(self,wo, W_all_y):
        """

        :param wo: [b_s,d_c, 1]
        :param W_all_y: [b_s,num_rel,rel_emb ]
        :return:
        """
        wo_norm = F.normalize(wo.squeeze(dim=2)) # [b_s,d_c]
        wo_norm_tile = wo_norm.unsqueeze(1).repeat(1,W_all_y.size()[0],1)   #[b_s,num_rel,d_c ]
        dist = torch.norm(wo_norm_tile-W_all_y, 2, 2)
        predict = torch.min(dist, 1)[1].long()  # [bs, 1]
        return predict

    def forward(self, x_in, pos1, pos2, e1_vec, e2_vec, rel_vec):
        """

        :param x_in: sentence
        :param pos1:
        :param pos2:
        :param e1_vec:
        :param e2_vec:
        :param rel_vec:
        :return: wo:[b_s,d_c, 1] ; predict:[b_s, 1]
        """
        embedding, word, entity1, entity2 = self.emb(x_in, pos1, pos2, e1_vec, e2_vec)
        Z = self.slide.createWin(embedding)
        R=self.atten_input( entity1, entity2, word, Z)
        R_star = self.conv(R)
        W_L,rel_emb_weight, W_all_y = self.selector(rel_vec, self.all_y)
        self.rel_emb_weight=rel_emb_weight
        wo = self.atten_pool(R_star, W_L)   #[b_s,d_c, 1]
        predict = self.predict(wo, W_all_y)
        return wo, predict, rel_emb_weight




