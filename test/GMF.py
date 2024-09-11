import scipy.sparse as sp
import numpy as np

# filename为test.rating的数据   类似于测试集的制作
def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])     # 用户名 电影名
            line = f.readline()
    return ratingList

# test.negative
def load_negative_file(filename):
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1:]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList


def load_rating_file_as_matrix(filename):
    """
    Read .rating file and Return dok matrix.
    The first line of .rating file is: num_users\t num_items
    """
    # Get number of users and items
    num_users, num_items = 0, 0   # 这俩记录用户编号和物品编号里面的最大值
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
    # Construct matrix
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)  # dok_matrix可以高效地逐渐构造稀疏矩阵。 存储是稀疏存储 toarray()
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if rating > 0:
                mat[user, item] = 1.0
            line = f.readline()
    return mat   # 0,1矩阵， 如果评过分就是1， 否则是0

class Dataset():
    def __init__(self, path):
        self.trainMatrix = load_rating_file_as_matrix(path+'.train.rating')
        self.testRatings = load_rating_file_as_list(path+'.test.rating')
        self.testNegatives = load_negative_file(path+'.test.negative')
        assert len(self.testRatings) == len(self.testNegatives)
    
    def Getdataset(self):
        return (self.trainMatrix, self.testRatings, self.testNegatives)
    
    # 开始导入原数据并进行处理
path = '/home/august/code/github/AI-RecommenderSystem/Recall/NeuralCF/Data/ml-1m'
dataset = Dataset(path)

train, testRatings, testNegatives = dataset.Getdataset()

import datetime
import numpy as np
import pandas as pd
from collections import Counter
import heapq

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torchkeras import summary, Model

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 一些超参数设置
topK = 10
num_factors = 8
num_negatives = 4
batch_size = 64
lr = 0.001

# 数据在processed Data里面
# trainA = np.load('/home/august/code/github/AI-RecommenderSystem/Recall/NeuralCF/ProcessedData/train.npy', allow_pickle=True)
# train = trainA.tolist()
# print(type(trainA))         # 输出: <class 'numpy.ndarray'>
# print(type(train))    # 输出: <class 'list'>
# testRatings = np.load('/home/august/code/github/AI-RecommenderSystem/Recall/NeuralCF/ProcessedData/testRatings.npy').tolist()
# testNegatives = np.load('/home/august/code/github/AI-RecommenderSystem/Recall/NeuralCF/ProcessedData/testNegatives.npy').tolist()

num_users, num_items = train.shape


# 制作数据   用户打过分的为正样本， 用户没打分的为负样本， 负样本这里采用的采样的方式
def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_items = train.shape[1]
    for (u, i) in train.keys():  # train.keys()是打分的用户和商品  u和i都是下表，u是用户的idx，i是物品的idx     
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        
        # negative instance
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            #print(u, j)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

user_input, item_input, labels = get_train_instances(train, num_negatives)

train_x = np.vstack([user_input, item_input]).T
print(train_x.shape)
print(train_x[1])
labels = np.array(labels)

# 构建成Dataset和DataLoader
train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(labels).float())
dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 测试一下
for (x, y) in iter(dl_train):
    print(x, y)
    break

class GMF(nn.Module):
    
    def __init__(self, num_users, num_items, latent_dim, regs=[0, 0]):
        super(GMF, self).__init__()
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim)
        self.linear = nn.Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs):
        # 这个inputs是一个批次的数据， 所以后面的操作切记写成inputs[0], [1]这种， 这是针对某个样本了， 我们都是对列进行的操作
        
        # 先把输入转成long类型
        inputs = inputs.long()
        
        # 用户和物品的embedding
        # inputs[:, 0]表示第0列 inputs[:, 1]表示第1列，它们都是size=64的一维数组
        MF_Embedding_User = self.MF_Embedding_User(inputs[:, 0])  # 这里踩了个坑， 千万不要写成[0]， 我们这里是第0列
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[:, 1])
        
        # 两个隐向量对位相乘
        predict_vec = torch.mul(MF_Embedding_User, MF_Embedding_Item) #torch.Size([64, 8])
        print(predict_vec.shape)
        # liner
        linear = self.linear(predict_vec) #torch.Size([64, 1])
        output = self.sigmoid(linear) #torch.Size([64, 1])
        print(output.shape)

        return output
    

# 看一下这个网络
model = GMF(1, 1, 10)
print(model)
# summary(model, input_shape=(2,))

## 设置
model = GMF(num_users, num_items, num_factors)
model.to(device)

# 简单测试一下模型
for (x, y) in iter(dl_train):
    x = x.cuda()
    print(model(x))
    break

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

# HitRation
def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

# NDCG
def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return np.log(2) / np.log(i+2)
    return 0

def eval_one_rating(idx):   # 一次评分预测
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')
    
    test_data = torch.tensor(np.vstack([users, np.array(items)]).T).to(device)
    predictions = _model(test_data)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i].data.cpu().numpy()[0]
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=lambda k: map_item_score[k])  # heapq是堆排序算法， 取前K个
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return hr, ndcg

def evaluate_model(model, testRatings, testNegatives, K):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    
    _model = model
    _testNegatives = testNegatives
    _testRatings = testRatings
    _K = K
    
    hits, ndcgs = [], []
    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return hits, ndcgs   
# 训练参数设置
loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

# 计算出初始的评估
(hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK)
hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
print('Init: HR=%.4f, NDCG=%.4f' %(hr, ndcg))

# 模型训练 
best_hr, best_ndcg, best_iter = hr, ndcg, -1

epochs = 20
log_step_freq = 10000

for epoch in range(epochs):
    
    # 训练阶段
    model.train()
    loss_sum = 0.0
    for step, (features, labels) in enumerate(dl_train, 1):
        
        features, labels = features.cuda(), labels.cuda()
        # 梯度清零
        optimizer.zero_grad()
        
        # 正向传播
        predictions = model(features)
        loss = loss_func(predictions.squeeze(-1), labels)
        
        # 反向传播求梯度
        loss.backward()
        optimizer.step()
        
        # 打印batch级别日志
        loss_sum += loss.item()
        if step % log_step_freq == 0:
            print(("[step = %d] loss: %.3f") %
                  (step, loss_sum/step))
    
    # 验证阶段
    model.eval()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    if hr > best_hr:
        best_hr, best_ndcg, best_iter = hr, ndcg, epoch
        torch.save(model.state_dict(), 'Pre_train/m1-1m_GMF.pkl')  
        
    info = (epoch, loss_sum/step, hr, ndcg)
    print(("\nEPOCH = %d, loss = %.3f, hr = %.3f, ndcg = %.3f") %info)
print('Finished Training...') 