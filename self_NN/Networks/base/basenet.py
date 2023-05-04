import torch
import torch.nn as nn
from collections import OrderedDict

class BaseNet(nn.Module):
    def __init__(self, config):
        super(BaseNet, self).__init__()
        self.config = config
        self.device = config.device

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])

    def forward(self, *inputs):
        """
        Defines the computation performed at every cell
        Inherited from Superclass torch.nn.Module
        Should be overridden by all subclasses
        :param inputs:
        :return:
        """
        raise NotImplementedError

    def get_inputs_(self, batch, **kwargs):
        """
        Define how to parse the data batch for the network training/testing
        This allows the flexibility for different training data formats in different applications
        Should be overridden by all subclasses
        :param batch:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def init_weights_(self):
        """
        Define how to initialize the weights of the network.
        Should be overridden by all suvclasses, since it normally
        differs according to the network models
        :return:
        """
        raise NotImplementedError

    def loss_(self, batch):
        """
        Define how to calculate the loss for the network.
        Should be overridden by all subclasses, since different applications
        or networks models may have different types of targets and the
        corresponding criterions to evaluate the predictions
        :param batch:
        :return:
        """
        raise  NotImplementedError

    def set_optimizer_(self, config):
        if config.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr_init, eps=config.epsilon, weight_decay=config.weight_decay)
        elif config.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=config.lr_init, momentum=config.momentum, weight_decay=config.weight_decay)

        # Initialize optimizer from a state if available
        if config.optimizer_dict and config.training:
            self.optimizer.load_state_dict(config.optimizer_dict)

        if config.lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay_factor, last_epoch= config.start_epoch-1)
        else:
            self.lr_scheduler = None


    """
    def my_func(*args):
        print(args)

    my_func(1, 2, 3) # 输出 (1, 2, 3)
    下一行
    def my_func(a, b, c):
        print(a, b, c)

    my_list = [1, 2, 3]
    my_func(*my_list) # 等同于 my_func(1, 2, 3)
    """
    def predict_(self, batch):
        return self.forward(*self.get_inputs_(batch, with_label=False))
    def optim_step_(self, batch):
        self.optimizer.zero_grad()
        loss, losses = self.loss_(batch)
        loss.backward()
        self.optimizer.step()
        return loss, losses

    def train_epoch(self, data_loader, epoch):
        if self.lr_scheduler:
            self.lr_scheduler.step()
        for i, batch in enumerate(data_loader):
            loss, losses = self.optim_step_(batch)
        # 这个地方很奇怪，不知道为什么是+，总感觉是+
        return loss, losses

    def save_weights_(self, sav_path):
        torch.save(self.state_dict(), sav_path)

    def print_(self):
        for k, v in self.state_dict().items():
            print(k, v.size())

    def kaiming_normal_init_func_(self, m):
        classname = m.__class__.__name__
        '''
        Not finised
        '''

    def xavier_init_func_(self, m):
        classname = m.__class__.__name__
        '''
        Not finised
        '''
    def normal_init_func(self, m):
        classname = m.__class__.__name__
        '''
        Not finised
        '''

    def fixed_weighting_loss(self, loss_pos, loss_rot, beta=1.0):
        return loss_pos + beta * loss_rot

    def learned_weighted_loss(self, loss_pos, loss_rot, sx, sq):
        return (-1 * sx).exp() * loss_pos + sx +(-1 * sq).exp() * loss_rot + sq