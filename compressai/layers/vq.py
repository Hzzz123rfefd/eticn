import numpy as np
import torch
import torch.nn as nn

"""
    init:
        feather_shape(C,H,W): shape of feather
        codebook_size(K) : size of code book
    

    input:
        feather(C,H,W):latent vector(feather)

    output:
        decoder_input(C,H,W):reconstruction latent 
        zq(C,H,W):reconstruction latent 
        nearest_neighbor(1,H,W):index vector of latent vector
        embedding = self.embedding.weight.data(K,C):code book

"""



class VectorQuantizer(nn.Module):
    def __init__(self,codebook_dim,codebook_size):
        super(VectorQuantizer, self).__init__()
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        # codebook
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        self.embedding.weight.data.uniform_(-1./codebook_size, 1./codebook_size)

    def forward(self,feather):
        embedding = self.embedding.weight.data
        N, C, H, W = feather.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = feather.reshape(N, 1, C, H, W)
        # 计算每个像素点和K个码子的距离  （b，k，h，w）
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)  
        # 选择距离最近的码子索引        （b，h，w）
        nearest_neighbor = torch.argmin(distance, 1) 
        # make C to the second dim
        zq = self.embedding(nearest_neighbor).permute(0, 3, 1, 2)
        # stop gradient
        decoder_input = feather + (zq - feather).detach()
        return decoder_input,zq,nearest_neighbor.unsqueeze(1)
    
    def to_index(self,feather):
        #self.embedding.weight.data[0] = torch.zeros(self.codebook_dim).to(self.embedding.weight.data.device)
        embedding = self.embedding.weight.data
        N, C, H, W = feather.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = feather.reshape(N, 1, C, H, W)
        # 计算每个像素点和K个码子的距离  （b，k，h，w）
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)  
        # 选择距离最近的码子索引        （b，h，w）
        nearest_neighbor = torch.argmin(distance, 1) 
        return nearest_neighbor.unsqueeze(1)

    def to_feather(self,index):
        feather = self.embedding(index).permute(0, 3, 1, 2)
        return feather
    
class ChannelGroupVectorQuantizer(nn.Module):
    def __init__(self,embedding_dim,group_num,codebook_size):
        super(ChannelGroupVectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.group_num = group_num
        self.group_size = (int)(embedding_dim/group_num)
        self.codebook_size = codebook_size

        self.vector_quantization_groups = nn.ModuleList([
            VectorQuantizer(codebook_dim = 
                (int)(embedding_dim/group_num),
                codebook_size = codebook_size
            ) 
            for _ in range(group_num)
        ])

    def forward(self,feather):
        groups = [feather[:, i*self.group_size:(i+1)*self.group_size, :, :] for i in range(self.group_num)]
        for i in range(self.group_num):
            if i == 0:
                reconstruction_feather,zq,code_index = self.vector_quantization_groups[i](groups[i])
            else:
                tr,tz,tc = self.vector_quantization_groups[i](groups[i])
                reconstruction_feather = torch.cat((reconstruction_feather, tr), dim=1)
                zq = torch.cat((zq, tz), dim=1)
                code_index = torch.cat((code_index,tc),dim=1)
        return reconstruction_feather,zq,code_index
    
    def to_index(self,feather):
        groups = [feather[:, i*self.group_size:(i+1)*self.group_size, :, :] for i in range(self.group_num)]
        for i in range(self.group_num):
            if i == 0:
                code_index = self.vector_quantization_groups[i].to_index(groups[i])
            else:
                tc = self.vector_quantization_groups[i].to_index(groups[i])
                code_index = torch.cat((code_index,tc),dim=1)
        return code_index

    def to_feather(self,index):
        indexs = [index[:,i,:,:] for i in range(self.group_num)]
        for i in range(self.group_num):
            if i == 0:
                feather = self.vector_quantization_groups[i].to_feather(indexs[i])
            else:
                tf = self.vector_quantization_groups[i].to_feather(indexs[i])
                feather = torch.cat((feather,tf),dim=1)

        return feather