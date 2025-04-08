import random
from similarity_methods import *
from model.demo1 import *
from model.GAT import *
from model.MLP import *
from model.FKAN import *
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, matthews_corrcoef# 读入数据文件

# 读入数据文件

A = np.loadtxt("./data/MDAD/drug_microbe_matrix.txt" )  # Adjacency matrx

mis = np.loadtxt('./data/MDAD/microbe_similarity.txt')
drs = np.loadtxt('./data/MDAD/drug_similarity.txt')

known = np.loadtxt("./data/MDAD/known.txt")  # 已知关联索引
unknown = np.loadtxt("./data/MDAD/unknown.txt")  # 未知关联索引

adj = np.loadtxt("./data/MDAD/adj21.txt")
ptr = np.loadtxt('./data/MDAD/ptr.txt')
labels = np.loadtxt('./data/MDAD/adj.txt')


# SM:基于药物微生物关联的药物相似性矩阵，
# 重启随机游走算法，估计两个节点之间的接近度
def RWR(SM):
    alpha = 0.1
    E = np.identity(len(SM))  # 单位矩阵
    M = np.zeros((len(SM), len(SM)))
    s = []
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


def Net_construct(Sr_m, Sm_r):  # 异构网络的构建
    N1 = np.hstack((Sr_m, A))
    N2 = np.hstack((A.T, Sm_r))
    Net = np.vstack((N1, N2))
    return Net


all_auc = []
all_aupr = []
scores = []
drug_index = []
tables = []

temp_label = np.zeros((1373, 173))
for temp in labels:
    temp_label[int(temp[0]) - 1, int(temp[1] - 1)] = int(temp[2])
labels = temp_label


# 5-fold cv

def kflod_5(n):
    k = []
    unk = []

    lk = len(known)
    luk = len(unknown)
    for i in range(lk):
        k.append(i)
    for i in range(luk):
        unk.append(i)
    random.shuffle(k)
    random.shuffle(unk)

    num_test = int(np.floor(labels.shape[0] / 5))
    num_train = labels.shape[0] - num_test

    all_index = list(range(labels.shape[0]))
    np.random.shuffle(all_index)

    for cv in range(1, 6):
        interaction = np.array(list(A))
        if cv < 5:
            B1 = known[k[(cv - 1) * (lk // 5):(lk // 5) * cv], :]  # 1/5的1的索引
            B2 = unknown[unk[(cv - 1) * (luk // 5):(luk // 5) * cv], :]  # 1/5的0的索引
            for i in range(lk // 5):
                interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
        else:
            B1 = known[k[(cv - 1) * (lk // 5):lk], :]
            B2 = unknown[unk[(cv - 1) * (luk // 5):luk], :]
            for i in range(lk - (lk // 5) * 4):
                interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0

        Sm_r_GIP = GIP_Calculate(interaction)  # 微生物GIP相似性
        Sr_m_GIP = GIP_Calculate1(interaction)  # 药物GIP相似性

        Sr_m = []
        Sm_r = []
        num1 = 0
        for i in drs:
            num2 = 0
            for j in i:
                if j != 0:
                    Sr_m.append((j + Sr_m_GIP[num1][num2]) / 2)
                else:
                    Sr_m.append(Sr_m_GIP[num1][num2])
                num2 += 1
            num1 += 1
        Sr_m = np.array(Sr_m)
        Sr_m = np.reshape(Sr_m, (1373, 1373))

        num3 = 0
        for i in mis:
            num4 = 0
            for j in i:
                if j != 0:
                    Sm_r.append((j + Sm_r_GIP[num3][num4]) / 2)
                else:
                    Sm_r.append(Sm_r_GIP[num3][num4])
                num4 += 1
            num3 += 1
        Sm_r = np.array(Sm_r)
        Sm_r = np.reshape(Sm_r, (173, 173))
        Srr1 = RWR(Sr_m)
        Smm1 = RWR(Sm_r)


        Net = Net_construct(Srr1, Smm1)  # 异构网络1


        Sm_r_Sim = Cosine_Sim(mis)
        Sr_m_Sim = Cosine_Sim(drs)

        Feature = np.vstack((np.hstack((Sr_m_Sim, np.zeros((1373, 173)))),
                             np.hstack((np.zeros((173, 1373)), Sm_r_Sim))))

        Feature1 = train22(Feature, Net)
        Feature3 = train(Net)
        SF = np.hstack((Feature3, Feature1))


        print("------------------------------------------")
        train_indx = all_index[:num_train]
        test_idx = all_index[num_train:(num_train + num_test)]
        Y_train = labels[train_indx]
        Y_test = labels[test_idx]
        X_train = SF[train_indx]
        X_test = SF[test_idx]
        y_score = train_mlp(X_train, Y_train, X_test, Y_test)


        y_pred = [item for item in y_score.flatten()]
        # auc计算
        fpr, tpr, threshold = roc_curve(Y_test.flatten(), y_pred)
        auc_val = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(Y_test.flatten(), y_pred)
        average_precision = average_precision_score(Y_test.flatten(), y_pred)
        aupr = auc(recall, precision)

        print("average_precision:{}".format(average_precision))
        print("auc_val:{},aupr:{}".format(auc_val, aupr))
        print("fold cv--{}".format(cv))
        np.savetxt("./fpr" + str(n) + ".txt", fpr)
        np.savetxt("./tpr" + str(n) + ".txt", tpr)
        np.savetxt("./precision" + str(n) + ".txt", precision)
        np.savetxt("./recall" + str(n) + ".txt", recall)

    return auc_val



auc_val = []
aupr = []
for i in range(1,6):  # 5次的5折交叉验证
    a = kflod_5(i)
    print("------------------------------")
    auc_val.append(a)
    np.savetxt("./data/auc.txt", auc_val)
print(np.mean(auc_val))
print("------------------------------")
max_value = max(auc_val)
print(auc_val.index(max_value)+1)
