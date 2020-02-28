import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
class DistanceLoss(nn.Module):
    def __init__(self, margin,rel_emb_size, rel_vocab_size,all_y, padding_idx=0):
        super(DistanceLoss, self).__init__()
        self._margin = margin
        self.all_y = torch.from_numpy(np.array(all_y))
        self.relemb = nn.Embedding(embedding_dim=rel_emb_size,
                                   num_embeddings=rel_vocab_size,
                                   padding_idx=padding_idx)
        self.relemb.weight.requires_grad=False

    def forward(self, wo, rel_weight, in_y):
        """

        :param wo:[b_s,d_c, 1]
        :param rel_weight:
        :param in_y: [b_s, 1]
        :return:
        """
        self.relemb.weight=nn.Parameter(rel_weight)
        wo_norm = F.normalize(wo.squeeze())     #[b_s,d_c]
        wo_norm_tile = wo_norm.unsqueeze(1).repeat(1, self.all_y.shape[0], 1)  # [b_s, num_rel, d_c]
        rel_emb = self.relemb(in_y).squeeze(dim=1)  # [b_s, rel_emb]
        all_y_emb = self.relemb(self.all_y)     # [b_s, num_rel, rel_emb]
        y_dist=torch.norm(wo_norm - rel_emb, 2, 1) # [b_s, rel_emb]
        # 求最大的错分距离
        # Mask in_y
        all_dist = torch.norm(wo_norm_tile - all_y_emb, 2, 2)       # [b_s, num_rel, rel_emb]

        one_hot_y = torch.zeros(in_y.shape[0],self.all_y.shape[0]).scatter_(1, in_y, 1)
        masking_y = torch.mul(one_hot_y, 10000)
        _t_dist = torch.min(torch.add(all_dist, masking_y), 1)[0]
        loss = torch.mean(self._margin + y_dist - _t_dist)
        return loss
