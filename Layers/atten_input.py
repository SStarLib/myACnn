import torch.nn as nn
import torch
import torch.nn.functional as F
class Atten_input(nn.Module):
    def __init__(self, config):
        super(Atten_input, self).__init__()
        self.config = config
    def forward(self, entity1, entity2, word, z):
        """
        b_s:batch_size, s_l:length of sentence,e_s:embedding size
        e(n)_l: length of entity n n={1, 2}, pe_s: position embedding size
        :param entity1: [b_s, e1_l, e_s]
        :param entity2: [b_s, e2_l, e_s]
        :param word:shape: [b_s, s_l, e_s]
        :param z: [b_s, s_l, e_s+2*pe_s]
        :return:
        """
        # mean or sum ???
        # 此处出错，应该用 似乎用 w_d 点乘 e
        # 此处出错，mean（）会减去一个维度, mean(dim=2)
        A1 = torch.bmm(word, entity1.permute(0,2,1)).mean(dim=2) # [b_s, s_l, 1]
        A2 = torch.bmm(word, entity2.permute(0,2,1)).mean(dim=2) # [b_s, s_l, 1]
        alpha1 = F.softmax(A1, dim=1).unsqueeze(dim=2)   # [b_s, s_l, 1]
        alpha2 = F.softmax(A2, dim=1).unsqueeze(dim=2)   # [b_s, s_l, 1]
        R = torch.mul(z, (alpha1+alpha2)/2) # [b_s, s_l, e_s+2*pe_s]
        return R