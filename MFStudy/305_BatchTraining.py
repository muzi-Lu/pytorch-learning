# MiniBatchTraining

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

BATCH_SIZE = 4

# Fake Data
x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = TensorDataset(x, y)
loader = DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=False) # 为什么在这里不写dataset进Dataloader里面

for epoch in range(10):
    for step, (batch_x, batch_y) in enumerate(loader):
        # Training
        print('Epoch:', epoch+1, '|Step:', step+1, '|batch x:',
              batch_x.numpy(), '|batch y:', batch_y.numpy())



