import torch.nn as nn
import torch.nn.functional as F
import torch


class GMF(nn.Module):

    def __init__(self, num_users, num_items, latent_dim, regs=[0, 0]):
        super(GMF, self).__init__()
        self.MF_Embedding_User = nn.Embedding(
            num_embeddings=num_users, embedding_dim=latent_dim)
        self.MF_Embedding_Item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=latent_dim)
        self.linear = nn.Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # 这个inputs是一个批次的数据， 所以后面的操作切记写成inputs[0], [1]这种， 这是针对某个样本了， 我们都是对列进行的操作

        # 先把输入转成long类型
        inputs = inputs.long()

        # 用户和物品的embedding
        # inputs[:, 0]表示第0列 inputs[:, 1]表示第1列，它们都是size=64的一维数组
        MF_Embedding_User = self.MF_Embedding_User(
            inputs[:, 0])  # 这里踩了个坑， 千万不要写成[0]， 我们这里是第0列
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[:, 1])

        # 两个隐向量对位相乘
        # torch.Size([64, 8])
        predict_vec = torch.mul(MF_Embedding_User, MF_Embedding_Item)
        # print(predict_vec.shape)

        # liner
        linear = self.linear(predict_vec)  # torch.Size([64, 1])
        output = self.sigmoid(linear)  # torch.Size([64, 1])

        return output
    
# 看一下这个网络
# model = GMF(1, 1, 10)
# print(model)


class MLP(nn.Module):
    
    def __init__(self, num_users, num_items, layers=[20, 64, 32, 16], regs=[0, 0]):
        super(MLP, self).__init__()
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=layers[0]//2)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=layers[0]//2)
        
        # 全连接网络
        self.dnn_network = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        self.linear = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs):
        # 这个inputs是一个批次的数据， 所以后面的操作切记写成inputs[0], [1]这种， 这是针对某个样本了， 我们都是对列进行的操作
        # 先把输入转成long类型
        inputs = inputs.long()
        
        # MF的前向传播  用户和物品的embedding
        MF_Embedding_User = self.MF_Embedding_User(inputs[:, 0])  # 这里踩了个坑， 千万不要写成[0]， 我们这里是第一列
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[:, 1])
        
        
        # 两个隐向量堆叠起来
        x = torch.cat([MF_Embedding_User, MF_Embedding_Item], dim=-1)
        
        # l全连接网络
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        
        x = self.linear(x)
        output = self.sigmoid(x)
        
        return output


class NeuralMF(nn.Module):
    
    def __init__(self, num_users, num_items, mf_dim, layers=[20, 64, 32, 16]):
        super(NeuralMF, self).__init__()
        
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=mf_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=mf_dim)
        
        self.MLP_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=layers[0] // 2)
        self.MLP_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=layers[0] // 2)
        
        # 全连接网络
        self.dnn_network = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        self.linear = nn.Linear(layers[-1], mf_dim)
        
        # 合并之后
        self.linear2 = nn.Linear(2*mf_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs):
        # 这个inputs是一个批次的数据， 所以后面的操作切记写成inputs[0], [1]这种， 这是针对某个样本了， 我们都是对列进行的操作
        
        # 先把输入转成long类型
        inputs = inputs.long()
        
        # MF模型的计算 用户和物品的embedding
        MF_Embedding_User = self.MF_Embedding_User(inputs[:, 0])  # 这里踩了个坑， 千万不要写成[0]， 我们这里是第一列
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[:, 1])
        # 两个向量点积过一个全连接
        mf_vec = torch.mul(MF_Embedding_User, MF_Embedding_Item)
        
        # MLP 模型的计算
        MLP_Embedding_User = self.MLP_Embedding_User(inputs[:, 0])  
        MLP_Embedding_Item = self.MLP_Embedding_Item(inputs[:, 1])
        # 两个隐向量堆叠起来
        x = torch.cat([MLP_Embedding_User, MLP_Embedding_Item], dim=-1)
        # l全连接网络
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        mlp_vec = self.linear(x)
        
        # 合并两个
        vector = torch.cat([mf_vec, mlp_vec], dim=-1)
        
        # liner
        linear = self.linear2(vector)
        output = self.sigmoid(linear)
        
        return output