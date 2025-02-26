import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from tools.tools import *

class MA_MSE(nn.Module):
    def __init__(self, fea_dim=20*128, n_heads=8, speaker_embedding_path=""):
        super(MA_MSE, self).__init__()
        self.n_heads = n_heads

        #Dictionary number of cluster * speaker_embedding_dim
        self.m = torch.from_numpy(np.load(speaker_embedding_path).astype(np.float32))
        self.N_clusters, Emb_dim = self.m.shape

        # Define matrices W (from audio feature) and U (from embedding)
        self.W = nn.Linear(fea_dim, n_heads)
        self.U = nn.Linear(Emb_dim, n_heads)

        self.v = nn.Linear(n_heads, 1)

    def forward(self, x, mask):
        '''
        x: Batch * Fea * Time
        mask: Batch * speaker * Time
        '''
        Batch, Fea, Time = x.shape
        num_speaker = mask.shape[1]
        #x_1: [Batch, num_speaker, Time, Fea]
        x_1 = x.repeat(1, num_speaker, 1).reshape(Batch, num_speaker, Fea, Time).transpose(2, 3)

        #x_2: Average [Batch, num_speaker, Fea]
        x_2 = torch.sum(x_1 * mask[..., None], axis=2) / (1e-10 + torch.sum(mask, axis=2)[..., None])

        #self.W(x_2) [Batch, num_speaker, n_heads]
        w = self.W(x_2).repeat(1, self.N_clusters, 1).reshape(Batch, self.N_clusters, num_speaker, self.n_heads).transpose(1, 2)

        #self.U(self.m) [N_clusters, n_heads]
        m = self.m.to(x.device)
        u = self.U(m).repeat(Batch*num_speaker, 1).reshape(Batch, num_speaker, self.N_clusters, self.n_heads)

        #c: Attention [Batch, num_speaker, N_clusters]
        c = self.v(torch.tanh(w + u)).squeeze(dim=3)

        #a: normalized attention values [Batch, num_speaker, N_clusters]
        a = torch.sigmoid(c)

        #e: weighted sum of the vectors [Batch, num_speaker, Emb_dim]
        #[Batch, num_speaker, N_clusters, 1] * [1, 1, N_clusters, Emb_dim]
        e = torch.sum(a[..., None] * m[None, None, ...], dim=2)

        return e


class MULTI_SE_MA_MSE_NSD(nn.Module):

    def __init__(self, configs):
        super(MULTI_SE_MA_MSE_NSD, self).__init__()
        self.input_size = configs["input_dim"]
        self.Linear_dim = configs["Linear_dim"]
        self.output_speaker = configs["output_speaker"]
        #self.batchnorm = nn.BatchNorm2d(1,device='cuda')
        self.average_pooling = nn.AvgPool1d(configs["average_pooling"], stride=2, padding=configs["average_pooling"]//2)
        # MA-MSE
        self.mamse1 = MA_MSE(fea_dim=configs["fea_dim"], n_heads=configs["n_heads1"], speaker_embedding_path=configs["embedding_path1"])
        self.mamse2 = MA_MSE(fea_dim=configs["fea_dim"], n_heads=configs["n_heads2"], speaker_embedding_path=configs["embedding_path2"])

        self.splice_size = configs["splice_size"]
        self.Linear = nn.Linear(self.splice_size, self.Linear_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x, overall_embedding, mask, split_seg=-1, return_embedding=False):
        '''
        x: Batch * Freq * Time
        mask : Batch * speaker * Time
        overall_embedding: Batch * speaker(4) * Embedding
        split: split long sequence to shorter segments to accelerate BLSTM training
                Time % split_seg == 0
        '''
        x = self.average_pooling(x)
        batchsize, Freq, Time = x.shape
        embedding1 = self.mamse1(x, mask) # [Batch, num_speaker, Emb_dim]
        embedding2 = self.mamse2(x, mask) # [Batch, num_speaker, Emb_dim]

        x_reshape = x.repeat(1, self.output_speaker, 1).reshape(batchsize * self.output_speaker, Freq, Time)
        #embedding: Batch * speaker * Embedding -> (Batch * speaker) * Embedding * Time
        embedding_dim1 = embedding1.shape[2]
        embedding_reshape1 = embedding1.reshape(-1, embedding_dim1)[..., None].expand(batchsize * self.output_speaker, embedding_dim1, Time)

        embedding_dim2 = embedding2.shape[2]
        embedding_reshape2 = embedding2.reshape(-1, embedding_dim2)[..., None].expand(batchsize * self.output_speaker, embedding_dim2, Time)

        overall_embedding_dim = overall_embedding.shape[2]
        overall_embedding_reshape = overall_embedding.reshape(-1, overall_embedding_dim)[..., None].expand(batchsize * self.output_speaker, overall_embedding_dim, Time)

        x_1 = torch.cat((x_reshape, embedding_reshape1, embedding_reshape2, overall_embedding_reshape), dim=1)
        '''
        x_1: (Batch * speaker) * (Freq + Embedding) * Time
        '''
        #**************Linear*************
        #(Batch * speaker) * (Freq + Embedding) * Time =>(Batch * speaker) * Time * Linear_dim
        x_2 = self.relu(self.Linear(x_1.transpose(1, 2)))
        return x_2