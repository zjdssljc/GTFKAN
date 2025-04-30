import pandas as pd
import numpy as np

A1 = np.loadtxt("data/MDAD/drug_disease_matrix.txt" )
A2 = np.loadtxt("data/MDAD/microbes_disease_matrix.txt" )

def calculate_gaussian_kernel_similarity(A, n, axis):
    """
    计算高斯核相似性
    :param A: 邻接矩阵
    :param n: 行数或列数
    :param axis: 0 表示按列计算（微生物相似性），1 表示按行计算（药物相似性）
    :return: 高斯核相似性矩阵
    """
    if axis == 1:  # 计算药物相似性
        L = A
        mu = 1 / (np.sum(np.linalg.norm(L, axis=1) ** 2) / n)
        GIR = np.exp(-mu * np.linalg.norm(L[:, np.newaxis] - L[np.newaxis, :], axis=2) ** 2)
        return GIR
    else:  # 计算微生物相似性
        M = A.T
        gamma = 1 / (np.sum(np.linalg.norm(M, axis=1) ** 2) / n)
        GIM = np.exp(-gamma * np.linalg.norm(M[:, np.newaxis] - M[np.newaxis, :], axis=2) ** 2)
        return GIM

def calculate_semantic_similarity(A):
    """
    计算语义相似性
    :param A: 邻接矩阵
    :return: 疾病语义相似性矩阵
    """
    D = np.where(A > 0, 1, 0)
    num_diseases = D.shape[1]
    DS = np.zeros((num_diseases, num_diseases))
    for i in range(num_diseases):
        D_i = np.where(D[:, i] > 0)[0]
        for j in range(num_diseases):
            D_j = np.where(D[:, j] > 0)[0]
            common = np.intersect1d(D_i, D_j)
            H_di = np.zeros(len(common))
            H_dj = np.zeros(len(common))
            for k in range(len(common)):
                H_di[k] = 1 if common[k] == i else 0.5
                H_dj[k] = 1 if common[k] == j else 0.5
            DS[i, j] = np.sum(H_di + H_dj) / (np.sum(D[:, i]) + np.sum(D[:, j]))
    return DS

def calculate_functional_similarity(A, DS, axis):
    """
    计算功能相似性
    :param A: 邻接矩阵
    :param DS: 疾病语义相似性矩阵
    :param axis: 0 表示计算微生物功能相似性，1 表示计算药物功能相似性
    :return: 功能相似性矩阵
    """
    if axis == 1:  # 计算药物功能相似性
        num_drugs = A.shape[0]
        FR = np.zeros((num_drugs, num_drugs))
        for i in range(num_drugs):
            d_i1 = np.where(A[i, :] > 0)[0]
            for j in range(num_drugs):
                d_j2 = np.where(A[j, :] > 0)[0]
                max_DE = np.zeros((len(d_i1), len(d_j2)))
                for k in range(len(d_i1)):
                    for l in range(len(d_j2)):
                        max_DE[k, l] = DS[d_i1[k], d_j2[l]]
                FR[i, j] = (np.max(max_DE, axis=1).sum() + np.max(max_DE, axis=0).sum()) / (len(d_i1) + len(d_j2))
        return FR
    else:  # 计算微生物功能相似性
        num_microbes = A.shape[0]
        FM = np.zeros((num_microbes, num_microbes))
        for i in range(num_microbes):
            d_i3 = np.where(A[i, :] > 0)[0]
            for j in range(num_microbes):
                d_j4 = np.where(A[j, :] > 0)[0]
                max_DE = np.zeros((len(d_i3), len(d_j4)))
                for k in range(len(d_i3)):
                    for l in range(len(d_j4)):
                        max_DE[k, l] = DS[d_i3[k], d_j4[l]]
                FM[i, j] = (np.max(max_DE, axis=1).sum() + np.max(max_DE, axis=0).sum()) / (len(d_i3) + len(d_j4))
        return FM

def calculate_integrated_similarity(GIR, FR, GIM, FM):
    """
    计算综合相似性
    :param GIR: 药物高斯核相似性矩阵
    :param FR: 药物功能相似性矩阵
    :param GIM: 微生物高斯核相似性矩阵
    :param FM: 微生物功能相似性矩阵
    :return: 药物综合相似性矩阵和微生物综合相似性矩阵
    """
    ISR = np.where(FR != 0, (GIR + FR) / 2, GIR)
    ISM = np.where(FM != 0, (GIM + FM) / 2, GIM)
    return ISR, ISM



# 计算药物高斯核相似性
GIR = calculate_gaussian_kernel_similarity(A1, A1.shape[0], axis=1)

# 计算微生物高斯核相似性
GIM = calculate_gaussian_kernel_similarity(A2, A2.shape[0], axis=0)

# 计算语义相似性
DS1 = calculate_semantic_similarity(A1)
DS2 = calculate_semantic_similarity(A2)

# 计算药物功能相似性
FR = calculate_functional_similarity(A1, DS1, axis=1)

# 计算微生物功能相似性
FM = calculate_functional_similarity(A2, DS2, axis=0)

# 计算综合相似性
ISR, ISM = calculate_integrated_similarity(GIR, FR, GIM, FM)
np.savetxt("data/MDAD/embedding.txt", ISR)
np.savetxt("data/MDAD/drug_similarity.txt", ISM)