from model import GMF
import time
from loguru import logger

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mmap
import json


def count_lines(file_path):
    with open(file_path, 'r+') as file:
        mmapped_file = mmap.mmap(file.fileno(), 0)
        line_count = 0
        while mmapped_file.readline():
            line_count += 1
    return line_count


def saveDict(file_path, data):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


fileName = r'data/data_user_item.csv'

total = count_lines(fileName)

df = pd.read_csv(fileName, sep=',', header=None, names=['memberId', 'productCode', 'thirdCategoryId', 'buyCount', 'purchaseNum'],
                 dtype={'memberId': int, 'productCode': str, 'thirdCategoryId': int, 'buyCount': int, 'purchaseNum': int})


memberId_column = df['memberId']
unique_memberId_column = memberId_column.drop_duplicates()
num_users = unique_memberId_column.shape[0]

memberId2Indexdict = {value: index for index,
                      value in enumerate(unique_memberId_column)}
assert num_users == len(memberId2Indexdict)
saveDict("Pre_data/memberId2Indexdict.json", memberId2Indexdict)
productCode_column = df['productCode']
unique_productCode_column = productCode_column.drop_duplicates()
num_items = unique_productCode_column.shape[0]

productCode2Indexdict = {value: index for index,
                         value in enumerate(unique_productCode_column)}
assert num_items == len(productCode2Indexdict)
saveDict("Pre_data/productCode2Indexdict.json", productCode2Indexdict)

logger.info("num_users is {}, num_items is {},totalSize is {}, maxBuyCount is {}",
            num_users, num_items, total, df['buyCount'].max())

torch.save(torch.Tensor([num_users, num_items, total]), 'Pre_data/param.pth')


user_input = torch.zeros((total))
item_input = torch.zeros((total))
values = torch.zeros((total))

for index, row in df.iterrows():
    u = memberId2Indexdict[row['memberId']]
    t = productCode2Indexdict[row['productCode']]
    v = row['buyCount']
    user_input[index] = u
    item_input[index] = t

    # values[index] = torch.sigmoid(torch.tensor(v-2))
    values[index] = v
# 由于模型要求v的值小于1，所以我们使用sigmoid函数。
# 原先使用的函数是：每个值除以最大值，但是最大值为2700，如果1/2700=0.00037，这个值太小了，会导致模型认为购买数为1的时候基本没有。
# value=1的时候，下面sigmoid的值为0.5。也就是说,模型的推荐值小于0.5，代表不推荐
# 除以2，让sigmod的曲线更加平滑
values = torch.sigmoid((values-1)/2)
logger.info("max value is {}, min value is {}",  values.max(), values.min())
assert values.shape[0] == total

train_x = torch.stack([user_input, item_input], dim=1)

logger.info("train_x's shape is {}", train_x.shape)

# 保存张量到文件
torch.save(train_x, 'Pre_data/key_tensor.pth')
torch.save(values, 'Pre_data/value_tensor.pth')


print("Tensor saved to tensor.pth")
