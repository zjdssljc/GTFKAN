import torch
from torch import nn
from model.layers import GraphAttention
import torch.nn.functional as F
from model.datadeal import *
import torch.optim as optim
import time





class GAT(nn.Module):
    def __init__(self,nfeat,nclass,dropout,alpha):
        super(GAT, self).__init__()
        self.gal1=GraphAttention(nfeat,nclass, dropout,alpha)

    def forward(self,x,adj):
        Z=self.gal1(x,adj)

        #nadj=Graph_update(Z,adj)
        a=Z.detach().numpy()
        np.savetxt("./embedding.txt",a)
        ZZ=torch.sigmoid(torch.matmul(Z,Z.T))
        return ZZ

def train33(Net):
    print('Graph Attention Networks')

    adj=torch.FloatTensor(Net)
    x=torch.FloatTensor(Net)
    idx_train = range(1300)
    idx_test = range(1300, 1500)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    model=GAT(x.shape[1],128,0.4,0.2)
    optimizer = optim.Adam(model.parameters(),
                           lr=0.01, weight_decay=5e-4)
    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()



        output = model(x, adj)
        loss_train = F.mse_loss(output[idx_train,:], adj[idx_train,:])
        loss_train.backward()
        optimizer.step()
        print('==>>> epoch: {}, train loss: {:.6f},Time:{:.2f}'.format(epoch + 1, loss_train.item(), time.time() - t))
        return loss_train

    def test():
        model.eval()

        output = model(x, adj)
        loss_test = F.mse_loss(output[idx_test,:], adj[idx_test,:])
        print("Test set results:",
              "loss= {:.5f}".format(loss_test.item()))

    t_total = time.time()
    for epoch in range(20):
        loss = train(epoch)
        # if loss < 0.1:
        #     break

    print("Optimization Finished!")
    print("Total time elapsed: {:.5f}s".format(time.time() - t_total))

    test()


