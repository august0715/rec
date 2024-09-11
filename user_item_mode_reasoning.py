from loguru import logger

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import json

from torch.utils.data import DataLoader, Dataset, TensorDataset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import GMF


def fromDict(file_path):
    with open(file_path, 'r') as json_file:
        data_loaded = json.load(json_file)
        return data_loaded


memberId2Indexdict = fromDict("Pre_data/memberId2Indexdict.json")

productCode2Indexdict = fromDict("Pre_data/productCode2Indexdict.json")

num_factors = 32
param = torch.load('Pre_data/param.pth').long()
num_users = param[0]
num_items = param[1]
total = param[2]

# 实例化模型
model = GMF(num_users, num_items, num_factors)
# 切换到评估模式
model.eval()
# 加载保存的模型参数
model.load_state_dict(torch.load('Pre_train/user_item.pkl'))

# memberId = '37471'
memberId = '4504452'
# 评估一个用户一个商品
userIndex = memberId2Indexdict[memberId]
itemIndex = productCode2Indexdict['PC20070800000121']

input = torch.tensor(([[userIndex, itemIndex]]))
v = model(input)
print(v)
print(v.shape)


# 评估一个用户对所有商品
userIndex = memberId2Indexdict[memberId]

to_predict_item = num_items
user_tensor = torch.full((to_predict_item,), userIndex)
item_tensor = torch.arange(to_predict_item).view(to_predict_item, )
input2 = torch.stack([user_tensor, item_tensor], dim=1)
v2 = model(input2)
indices = torch.where(v2 > 0.75)[0]
print("排序前的位置：", indices.shape, indices)

# 获取这些元素
filter_values = v2[indices].reshape(indices.shape[0],)
print("排序前的值：", filter_values.shape, filter_values)
r = torch.stack([indices, filter_values], dim=1)

print("排序前的值r：", r)

# 提取第二行的元素
second_col = r[:, 1]

# 对第二行的元素进行排序，并获取排序后的索引
sorted_values, sorted_indices = torch.sort(second_col, descending=True)

# 使用这些索引对整个张量的列进行重新排列
sorted_tensor = r[sorted_indices, :]

print("排序后的张量：")
print(sorted_tensor)

filterItem = sorted_tensor[:, 0].long()
print(filterItem)

reversed_dict = {value: key for key, value in productCode2Indexdict.items()}

i = 0
for row in sorted_tensor:
    if i > 256:
        break
    itemIndex = row[0].long().item()
    print(reversed_dict[itemIndex])
    i = i+1
