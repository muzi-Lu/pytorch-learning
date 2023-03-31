import torch
import numpy as np

data = np.arange(6)
print('data:', data)
tensor = torch.from_numpy(data)
print('tensor:', tensor)
data = data.reshape(2, 3)
print('reshaped data', data)
tensor = torch.from_numpy(data)
print('reshaped tensor', tensor)
data = tensor.numpy()
print('backed data', data)

# 数组好像和numpy还不一样
data_mat = [[1, 2, 3],
            [2, 3, 4]]
print(data_mat)
# 让数组的类型去numpy类型
data_mat = np.array(data_mat)
print(data_mat)
