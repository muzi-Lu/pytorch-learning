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

    # 显示化
    # print(x.shape, x.squeeze().shape)
    # plt.scatter(x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy())
    # plt.show()

    # 随机初始化参数
    w = torch.randn(1, 1).to(device)
    b = torch.zeros(1, 1).to(device)

    lr = 0.02

    for ii in range(500):
        # forward: 计算loss
        y_pred = x.mm(w) + b.expand_as(y) # X@W等价于x.mm(w)
        loss = 0.5 * (y_pred - y) ** 2
        loss = loss.mean()

        # backward:手动计算梯度
        dloss = 1
        dy_pred = dloss * (y_pred - y)

        dw = x.t().mm(dy_pred)
        db = dy_pred.sum()

        # 更新参数
        w.sub_(lr * dw)
        b.sub_(lr * db)

        if ii % 10 == 0:

            # 画图
            display.clear_output(wait=True)
            x = torch.arange(0, 6).view(-1, 1)
            print(x.float().mm(w))
            print(b.expand_as(x))
            y = x.float().mm(w) + b.expand_as(x)
            plt.plot(x.cpu().numpy(), y.cpu().numpy())

            x2, y2 = get_fake_data(batch_size=32)
            plt.scatter(x2.numpy(), y2.numpy())

            plt.xlim(0, 5)
            plt.ylim(0, 13)
            plt.show()
            plt.pause(1)
    print('w: ', w.item(), 'b: ', b.item())







