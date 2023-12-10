import torch
import torch.nn as nn
import numpy as np

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class MovementEncoder(nn.Module): # (247, 512, 512) -> 1,842,688 
    def __init__(self, input_size, hidden_size, output_size):
        print('input_size', input_size)
        super().__init__()
        self.main = nn.Sequential(                          # pc: Chan_out * (Area of kernel * Chan_in (+BN))
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),    # pc: 512 * (4 * 1 * 247 + 1) = 506,368
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),   # pc: 512 * (4 * 1 * 512 + 1) = 1,049,088
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # self.avg_pool = torch.mean(dim=2)                
        self.out_net = nn.Linear(output_size, output_size)  # pc: 512 * (512 + 1) = 262,656
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)        
        # print('inputs.shape', inputs.shape)     # torch.Size([128, 259, 24]) B, C, L
        outputs = self.main(inputs)
        outputs = outputs.permute(0, 2, 1)
        # print('outputs.shape', outputs.shape)   # torch.Size([128, 6, 512])
        # outputs = self.avg_pool(outputs)          # torch.Size([128, 1, 512])
                
        return self.out_net(outputs)


class MovementDecoder(nn.Module): # (512, 512, 251) -> 1,657,407
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1),   # pc: 512 * (4 * 1 * 512 + 1) = 1,049,088
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1),  # pc: 512 * (4 * 1 * 251 + 1) = 514,560
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)  # pc: 251 * (251 + 1) = 63,252

        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        # inputs = inputs.repeat(1, 6, 1)     # HACK: windows size is fixed as 24, and 24/4=6, now reverse it back
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)
