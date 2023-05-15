import torch


def linear_beta_schedule(timesteps):
    pass


def extract(a, t, x_shape):
    pass


def exists(x):
    '''
    判断这个这个变量存不存在
    :param x:
    :return:
    '''
    return x is not None


def default(t, *args, **kwargs):
    '''
    这个函数不知道干什么用的
    :param t:
    :param args:
    :param kwargs:
    :return:
    '''
    pass


def identity(t, *args, **kwargs):
    '''

    :param t:
    :param args:
    :param kwargs:
    :return:
    '''
    pass


def cycle(dl):
    '''
    这玩意怎么暂停呀
    :param dl:
    :return:
    '''
    pass


def has_int_squareroot(num):
    '''
    这个整数是不是有平方根
    :param num:
    :return:
    '''
    pass


def num_to_groups(num, divisor):
    '''

    :param num:
    :param divisor:
    :return:
    '''
    pass


# normalization functions

def normalize_to_neg_one_to_one(img):
    '''
    图片归一化
    :param img:
    :return:
    '''
    pass


def unnormalize_to_zerq_to_one(t):
    '''
    把变量归一化到0-1
    :param t:
    :return:
    '''
    pass

