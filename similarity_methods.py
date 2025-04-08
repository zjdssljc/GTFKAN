import numpy as np
import math
import scipy.sparse as sp
import torch as th


#衡量结点之间的相似度
def HIP_Calculate(M):
    l=len(M)
    cl=np.size(M,axis=1)
    SM=np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            dnum = 0
            for k in range(cl):
                if M[i][k]!=M[j][k]:
                    dnum=dnum+1
            SM[i][j]=1-dnum/cl
    return SM

def GIP_Calculate(M):
    l=np.size(M,axis=1)
    sm=[]
    m=np.zeros((l,l))
    #计算gama
    for i in range(l):
        tmp=(np.linalg.norm(M[:,i]))**2
        sm.append(tmp)
    gama=l/np.sum(sm)

    for i in range(l):
        for j in range(l):
            m[i,j]=np.exp(-gama*((np.linalg.norm(M[:,i]-M[:,j]))**2))
    return m


def GIP_Calculate1(M):
    l=np.size(M,axis=0)
    sm=[]
    m=np.zeros((l,l))
    km=np.zeros((l,l))

    for i in range(l):
        tmp=(np.linalg.norm(M[i,:]))**2
        sm.append(tmp)
    gama=l/np.sum(sm)

    for i in range(l):
        for j in range(l):
            m[i,j]=np.exp(-gama*((np.linalg.norm(M[i,:]-M[j,:]))**2))

    for i in range(l):
        for j in range(l):
            km[i,j]=1/(1+np.exp(-15*m[i,j]+math.log(9999)))
    return km

def Cosine_Sim(M):
    l=len(M)
    SM = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            v1=np.dot(M[i],M[j])
            v2=np.linalg.norm(M[i],ord=2)
            v3=np.linalg.norm(M[j],ord=2)
            if v2*v3==0:
                SM[i][j]=0
            else:
                SM[i][j]=v1/(v2*v3)
    return SM


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+sp.eye(adj.shape[0])
    return adj_normalized.tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


