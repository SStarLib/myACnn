import torch.nn as nn
import torch
import torch.nn.functional as F
class Attn_pool(nn.Module):
    def __init__(self,config):
        super(Attn_pool, self).__init__()
        self.config = config
        self.in_dim = self.config.hidden_size
        self.out_dim = 1
        self.U = nn.Parameter(torch.randn(self.in_dim, self.out_dim))
        self.kernel_size = self.config.max_sen_len
        self.max_pool = nn.MaxPool1d(self.kernel_size, 1)
    def forward(self, R_star, W_L):
        """

        :param R_star: [b_s,d_c, s_l]
        :param W_L: [b_s,1,rel_emb ]
        :return:
        """
        RU = torch.matmul(R_star.permute(0,2,1), self.U) # [b_s, s_l,1]
        G = torch.matmul(RU,W_L)   # [b_s, s_l,rel_emb]
        AP = F.softmax(G, dim=1)
        RA = torch.mul(R_star, AP.transpose(2, 1))  #[b_s,d_c, s_l]
        wo = self.max_pool(RA)  #[b_s,d_c, 1]
        return wo


