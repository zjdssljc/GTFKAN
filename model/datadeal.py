import numpy as np
import math
from utils import *
import pandas as pd



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
            SM[i][j]=1-dnum/cl             #HIP计算出来的相似矩阵
    return SM
def GIP_Calculate(M):     #计算微生物高斯核相似性
    l=np.size(M,axis=1)
    sm=[]
    m=np.zeros((l,l))
    for i in range(l):
        tmp=(np.linalg.norm(M[:,i]))**2
        sm.append(tmp)
    gama=l/np.sum(sm)
    for i in range(l):
        for j in range(l):
            m[i,j]=np.exp(-gama*((np.linalg.norm(M[:,i]-M[:,j]))**2))
    return m
def GIP_Calculate1(M):     #计算药物高斯核相似性
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
def drug_disease_microbe():
    A=np.loadtxt("../data/MDAD/drug_microbe_matrix.txt")
    dd=np.loadtxt("./data/MDAD/drug_with_disease.txt")
    md=np.loadtxt("./data/MDAD/microbe_with_disease.txt")
    ddict={}
    mdict={}
    for i in range(len(dd)):
        ddict[i]=dd[i][1]
    for i in range(len(md)):
        mdict[i]=md[i][1]
    for i in range(1,110):
        k1=[k for k,v in ddict.items() if v==i]
        k2=[k for k,v in mdict.items() if v==i]
        for p in range(len(k1)):
            for q in range(len(k2)):
                A[int(dd[k1[p]][0])-1][int(md[k2[q]][0])-1]=1
    np.savetxt("./data/MDAD/drug_dmicrobe_matrix.txt",A)   #通过drug-disease-microbe 增加drug-microbe association的矩阵

def RWR(SM):
    alpha = 0.1
    E = np.identity(len(SM))  # 单位矩阵
    M = np.zeros((len(SM), len(SM)))
    s=[]
    for i in range(len(M)):
        for j in range(len(M)):
            M[i][j] = SM[i][j] / (np.sum(SM[i, :]))
    for i in range(len(M)):
        e_i = E[i, :]
        p_i1 = np.copy(e_i)
        for j in range(10):
            p_i = np.copy(p_i1)
            p_i1 = alpha * (np.dot(M, p_i)) + (1 - alpha) * e_i
        s.append(p_i1)
    return s

def Nodepairs(r,m):
    F=[]
    for i in range(len(r)):
        for j in range(len(m)):
            F.append(np.vstack((r[i,:],m[j,:])))
    F=np.array(F)
    return F

def get_roc(pos_prob,y_true):
	pos=y_true[y_true==1]
	neg=y_true[y_true==0]
	y=y_true[pos_prob.argsort()[::-1]]
	tpr_all=[0];fpr_all=[0]
	tpr=0;fpr=0
	x_step=1/float(len(neg))
	y_step=1/float(len(pos))
	area=0
	for  i in range(len(pos_prob)):
		if y[i]==1:
			tpr+=y_step
		else:
			area+=tpr*x_step
	return  area

def konwn1():
    A1 = np.loadtxt("../data/MDAD/drug_dmicrobe_matrix.txt")
    known1=[]
    unknown1=[]
    for i in range(len(A1)):
        for j in range(np.size(A1,axis=1)):
            tmp=[]
            tmp1=[]
            if A1[i][j]==1:
                tmp.append(i+1)
                tmp.append(j+1)
                known1.append(tmp)
            else:
                tmp1.append(i+1)
                tmp1.append(j+1)
                unknown1.append(tmp1)
    known1=np.array(known1)
    unknown1=np.array(unknown1)
    np.savetxt("./data/MDAD/known1.txt",known1,fmt="%d")
    np.savetxt("./data/MDAD/unknown1.txt",unknown1, fmt="%d")

