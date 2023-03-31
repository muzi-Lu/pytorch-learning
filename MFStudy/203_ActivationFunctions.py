import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)
x = Variable(x)
x_np = x.data.numpy()

y_relu = F.relu(x).data.numpy()
y_sigmoid = F.relu(x).data.numpy()
# y_tanh = F.tanh(x).data.numpy() UserWarning: nn.functional.tanh is deprecated. Use torch.tanh instead.
#   warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

plt.figure(1, figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.xlabel('fake data')
plt.ylabel('relu')

plt.subplot(2, 2, 2)
plt.plot(x_np, y_sigmoid, c='blue', label='sigmoid')
plt.xlabel('fake data')
plt.ylabel('sigmoid')

plt.subplot(2, 2, 3)
plt.plot(x_np, y_tanh, c='green', label='tanh')
plt.xlabel('fake data')
plt.ylabel('tanh')

plt.subplot(2, 2, 4)
plt.plot(x_np, y_softplus, c='yellow', label='softplus')
plt.xlabel('fake data')
plt.ylabel('softplus')
plt.show()

