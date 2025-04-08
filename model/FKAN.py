import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import time




def train22(Feature,Net, epochs=20, batch_size=128):
    print('FourierKAN')

    # 转换为 PyTorch 张量
    X_train = torch.tensor(Feature, dtype=torch.float32)
    Y_train = torch.tensor(Net, dtype=torch.float32)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 定义 FKAN 模型
    model = FourierModel(X_train.shape[1],Y_train.shape[1])  # 定义模型结构
    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(epochs):
        time_epoch_start = time.time()
        model.train()
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
        print('==>>> epoch: {}, train loss: {:.6f},Time:{:.2f}'.format(epoch + 1, loss, time.time() - time_epoch_start))

    # 测试模型
    model.eval()
    with torch.no_grad():
        # Feature, _ = model(X_train).detach().cpu().numpy()
        Feature= model(X_train)
        Feature_numpy = Feature.detach().cpu().numpy()
    return Feature_numpy


class NaiveFourierKANLayer(torch.nn.Module):
    def __init__( self, inputdim, outdim, gridsize, addbias=True, smooth_initialization=False):
        super(NaiveFourierKANLayer,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        grid_norm_factor = (torch.arange(gridsize) + 1)**2 if smooth_initialization else np.sqrt(gridsize)

        self.fouriercoeffs = torch.nn.Parameter( torch.randn(2,outdim,inputdim,gridsize) /
                                                (np.sqrt(inputdim) * grid_norm_factor ) )
        if( self.addbias ):
            self.bias  = torch.nn.Parameter( torch.zeros(1,outdim))

    def forward(self,x):
        xshp = x.shape
        # print(xshp)
        outshape = xshp[0:-1]+(self.outdim,)
        # print(outshape)
        x = torch.reshape(x,(-1,self.inputdim))
        k = torch.reshape( torch.arange(1,self.gridsize+1,device=x.device),(1,1,1,self.gridsize))
        xrshp = torch.reshape(x,(x.shape[0],1,x.shape[1],1) )
        c = torch.cos( k*xrshp )
        s = torch.sin( k*xrshp )
        y =  torch.sum( c*self.fouriercoeffs[0:1],(-2,-1))
        y += torch.sum( s*self.fouriercoeffs[1:2],(-2,-1))
        if( self.addbias):
            y += self.bias
        y = torch.reshape( y, outshape)
        return y


class FourierModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FourierModel, self).__init__()
        self.kan_layer1 = NaiveFourierKANLayer(inputdim=in_dim, outdim=256, gridsize=16)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 216)
        self.fc5 = nn.Linear(216, out_dim)
        # self.mlp_classifier = MLPDecoder(out_dim, 128, 256, out_dim1)


    def forward(self, x):
        x = self.kan_layer1(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        # score = self.mlp_classifier(x)
        return x


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x