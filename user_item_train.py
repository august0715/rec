from model import GMF,MLP,NeuralMF
import time
from loguru import logger


import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
logger.info("device is {}", device)

param = torch.load('Pre_data/param.pth')
param = param.long()
num_users = param[0]
num_items = param[1]
total = param[2]

logger.info("num_users is {}, num_items is {}, totalSize is {}",
            num_users, num_items, total)

train_x = torch.load('Pre_data/key_tensor.pth')
values = torch.load('Pre_data/value_tensor.pth')

logger.info("train_x's shape is {}", train_x.shape)
print(train_x)
print(values)


# 一些超参数设置
num_factors = 8
lr = 0.01

# 设置
model = GMF(num_users, num_items, num_factors)
# model = MLP(num_users, num_items)
# model = NeuralMF(num_users, num_items, num_factors)

model.to(device)


# 训练参数设置
loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)


# 模型训练
features, labels = train_x.to(device), values.to(device)
epochs = 3000
start_time = time.time()
loss_sum = 0.0
for epoch in range(epochs):

    # 训练阶段
    model.train()

    # 梯度清零
    optimizer.zero_grad()
    # 正向传播
    predictions = model(features)
    loss = loss_func(predictions.squeeze(-1), labels)
    # 反向传播求梯度
    loss.backward()
    optimizer.step()
    # 打印batch级别日志
    # print(type(loss))
    loss_sum += loss.item()
    logger.info("epoch={},  loss={}, avgloss={}",
                epoch, loss.item(), loss_sum/(epoch+1))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Block executed in: {elapsed_time:.4f} seconds")
torch.save(model.state_dict(), 'Pre_train/user_item.pkl')
print('Finished Training...')
