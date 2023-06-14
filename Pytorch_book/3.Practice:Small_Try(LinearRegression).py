import torch
from matplotlib import pyplot as plt
from IPython import display

def get_fake_data(batch_size=8):
    # 产生随机数据：y=x*2+3，加上了一些噪声
    x = torch.randn(batch_size, 1, device=device) * 5
    y = x * 2 + 3 + torch.randn(batch_size, 1, device=device)
    return x, y


if "__main__" == __name__:

    # 选取运行设备
    device = torch.device('cuda:0')

    # 设置随机数种子，保证在不同电脑上运行时下面输出一致
    torch.manual_seed(1000)
    x, y = get_fake_data()
    print(x, x.shape)
    print(y)









