import numpy as np
from scipy.sparse import dok_matrix

# 创建一个5x5的稀疏矩阵
dok = dok_matrix((5, 5), dtype=np.float32)

# 设置一些非零元素
dok[0, 0] = 1
dok[1, 2] = 2
dok[3, 4] = 3

# 访问元素
print(dok[0, 0])  # 输出: 1.0
print(dok[1, 2])  # 输出: 2.0
print(dok[2, 2])  # 输出: 0.0

# 转换为其他稀疏矩阵格式
csr = dok.tocsr()  # 转换为 CSR 格式
print(csr)

# l = dok.tolist()
# print(l)


import torch

# 假设 user_input 和 item_input 是 PyTorch 张量
user_input = torch.tensor([0, 1, 2, 3])
item_input = torch.tensor([10, 11, 12, 13])

# 使用 torch.stack 进行堆叠，然后转置
train_x = torch.stack([user_input, item_input], dim=1)

print(train_x)


import torch

# 创建一个形状为 (10, 2) 的张量，填充一些示例数据
tensor = torch.tensor([
    [9, 2],
    [8, 5],
    [4, 7],
    [1, 3],
    [6, 10],
    [11, 0],
    [3, 6],
    [8, 2],
    [5, 9],
    [7, 4]
])

# 提取第二行的元素
second_col = tensor[:, 1]

# 对第二行的元素进行排序，并获取排序后的索引
sorted_values, sorted_indices = torch.sort(second_col)

# 使用这些索引对整个张量的列进行重新排列
sorted_tensor = tensor[sorted_indices,: ]

print("排序后的张量：")
print(sorted_tensor)
print("排序后每个元素的原始索引：")
print(sorted_indices)
