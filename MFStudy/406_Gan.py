import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# Define Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR_G = 0.001
LR_D = 0.001
N_IDEAS = 10
ART_COMPONENTS = 20
# PAINT = [np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)]
# print(len(PAINT))
# print(PAINT)
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)]) # not understand
# print(PAINT_POINTS.shape)
# show our beautiful painting range
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
# plt.legend(loc='upper right')
# plt.show()

def artist_works():
    a = np.random.uniform(1, 2, size=BATCH_SIZE)
    print('a: ', a.shape)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    print('a: ', a.shape)
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    print('paintings:', paintings.shape)
    paintings = torch.from_numpy(paintings).float()
    print('paintings_1:', paintings.shape)
    print('paintings1_dtype:', paintings.dtype)
    return paintings

G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS)
)

D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()

for step in range(10000):
    artist_paintings = artist_works()
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)
    G_paintings = G(G_ideas)
    prob_artist1 = D(G_paintings)
    G_loss = torch.mean(torch.log(1. - prob_artist1))
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    prob_artist0 = D(artist_paintings)
    prob_artist1 = D(G_paintings.detach())
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)  # reusing computational graph
    opt_D.step()

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                 fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=10);
        plt.draw();
        plt.pause(0.01)

plt.ioff()
plt.show()