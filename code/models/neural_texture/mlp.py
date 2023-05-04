import torch
from torch import nn
from models.core_layers import normalization, non_linearity
from models.core_modules.standard_block import Conv2dBlock, ConvTrans2dBlock


class MLP(nn.Module):
    def __init__(self, param, model_param):

        super(MLP, self).__init__()
        self.param = param
        self.n_featutres = model_param.n_max_features
        self.encoding = model_param.encoding
        self.noise = model_param.noise

        self.nf_out = model_param.shape_out[0][0]
        self.nf_in = model_param.shape_in[0][0]
        self.n_blocks = model_param.n_blocks
        self.bias = model_param.bias

        # self.first_conv = Conv2dBlock(self.nf_in, self.n_featutres, 1, 1, 0, None, model_param.non_linearity, model_param.dropout_ratio, bias=self.bias)
        self.first_conv =torch.nn.Conv2d(in_channels= self.nf_in,out_channels=self.n_featutres, kernel_size=1,stride=1,padding=0,bias=self.bias)


        self.res_blocks = nn.ModuleList()

        for idx in range(self.n_blocks):
            # block_i = Conv2dBlock(self.n_featutres, self.n_featutres, 1, 1, 0, None, model_param.non_linearity, model_param.dropout_ratio, bias=self.bias)
            block_i = torch.nn.Conv2d(in_channels= self.n_featutres,out_channels=self.n_featutres, kernel_size=1,stride=1,padding=0,bias=self.bias)
            self.res_blocks.append(block_i)

        # self.last_conv = Conv2dBlock(self.n_featutres, self.nf_out, 1, 1, 0, None, None, model_param.dropout_ratio, bias=self.bias)
        self.last_conv =torch.nn.Conv2d(in_channels= self.n_featutres,out_channels=self.nf_out, kernel_size=1,stride=1,padding=0,bias=self.bias)

    def forward(self, input):
        # print('*'*50,type(input),input.shape)
        input_z = self.first_conv(input)
        output = input_z
        for idx, block in enumerate(self.res_blocks):
            output = block(output)

        output = self.last_conv(output)

        return output
