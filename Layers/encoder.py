import torch.nn as nn
import torch
class Slide(object):
    def __init__(self, config):
        self.config =config
        self.win_size = self.config.win_size
    def createWin(self, input_vec):
        """
        [b_s, s_l, e_s+2*pe_s], k=win_size
        :param input_vec: [b_s, s_l, e_s+2*pe_s]
        :return:shape [b_s, s_l, (e_s+2*pe_s)*k]
        """
        n = self.win_size
        result = input_vec
        input_len = input_vec.shape[1]
        for i in range(1,n):
            input_temp = input_vec.narrow(1,i, input_len-i) # 长度应该是 input_len - i
            ten = torch.zeros((input_vec.shape[0],i, input_vec.shape[2]),dtype=torch.float)
            input_temp = torch.cat((input_temp, ten),dim=1)
            result=torch.cat((result,input_temp), dim=2)
        return result

class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        self.config =config
        self.in_height = self.config.kernel_size
        self.in_width = \
            (self.config.word_embed_size + 2 * self.config.pos_embed_size)*self.config.win_size
        self.stride = (1, 1)
        self.padding = (1, 0)
        self.kernel_size = (self.in_height,self.in_width)
        self.cnn = nn.Conv2d(1, 1000, self.kernel_size,self.stride,self.padding)
    def forward(self, R):
        """
        d_c =hidden_size= 1000
        :param R: shape: [b_s, s_l, e_s+2*pe_s]
        :return: R_star  shape: [b_s,d_c, s_l]
        """
        R_star = torch.tanh(self.cnn(R.unsqueeze(dim=1)).squeeze(-1))
        return R_star
