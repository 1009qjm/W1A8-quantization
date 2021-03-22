from quantization import *

class Net(nn.Module):
    def __init__(self, abits=8, wbits=1, q_type=1):
        super(Net, self).__init__()
        # model - A/W全量化(除输入、输出外)
        self.quan_model = nn.Sequential(
            #nn.Conv2d(1,8,kernel_size=3,stride=1,padding=1),
            QuanConv2d(1, 16, kernel_size=3, stride=1, padding=1, abits=abits, wbits=wbits,q_type=q_type),
            nn.MaxPool2d(kernel_size=2, stride=2),

            QuanConv2d(16, 32, kernel_size=3, stride=1, padding=1, abits=abits, wbits=wbits,q_type=q_type),
            QuanConv2d(32, 64, kernel_size=3, stride=1, padding=1, abits=abits, wbits=wbits,q_type=q_type),
            nn.MaxPool2d(kernel_size=2, stride=2),

            QuanConv2d(64, 10, kernel_size=3, stride=1, padding=1, abits=abits, wbits=wbits, q_type=q_type),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.quan_model(x)
        x = x.view(x.size(0), -1)
        return x