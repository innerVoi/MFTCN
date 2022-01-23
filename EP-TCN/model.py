from torch import nn
from tcn import TemporalConvNet

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        # print(y1.shape)
        # print(y1)
        y = self.linear(y1[:,:,-1])
        # print(y)
        s, b = y.shape
        return y.view(s, b, -1)
