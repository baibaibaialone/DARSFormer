import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import metrics
import torch
import torch.nn as nn
import dgl
from numpy.linalg import eig, eigh
import pandas as pd
def load_data(directory, random_seed):
    # D_SSM1 = np.loadtxt(directory + '/D_SSM1.txt')
    # # WHAT??
    # D_SSM2 = np.loadtxt(directory + '/D_SSM2.txt')
    # D_GSM = np.loadtxt(directory + '/D_GSM.txt')
    # M_FSM = np.loadtxt(directory + '/M_FSM.txt')
    # M_GSM = np.loadtxt(directory + '/M_GSM.txt')
    # all_associations = pd.read_csv(directory + '/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label'])
    # ID = D_GSM
    # IM = M_GSM
    D_SSM1 = np.loadtxt(directory + '/D_SSM1.txt')
    D_SSM2 = np.loadtxt(directory + '/D_SSM2.txt')
    D_GSM = np.loadtxt(directory + '/D_GSM.txt')
    M_FSM = np.loadtxt(directory + '/M_FSM.txt')
    M_GSM = np.loadtxt(directory + '/M_GSM.txt')
    all_associations = pd.read_csv(directory + '/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label'])

    D_SSM = (D_SSM1 + D_SSM2) / 2
    ID = D_GSM
    IM = M_GSM
    # for i in range(D_SSM.shape[0]):
    #     for j in range(D_SSM.shape[1]):
    #         if ID[i][j] == 0:
    #             ID[i][j] = D_GSM[i][j]
    #
    # for i in range(M_FSM.shape[0]):
    #     for j in range(M_FSM.shape[1]):
    #         if IM[i][j] == 0:
    #             IM[i][j] = M_GSM[i][j]





    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)

    samples = sample_df.values      # 获得重新编号的新样本


    return ID, IM, samples      # 未知关联数量较多，选择和已知关联数目相同的未知关联组成样本


def build_graph(directory, random_seed):
    ID, IM, samples = load_data(directory, random_seed)


    # print(adj)
    # k=0
    # for i in range(adj.shape[0]):
    #     if(adj[0][i]==1):
    #         k=k+1
    # print(k)
    # miRNA和disease二元异质图
    g = dgl.DGLGraph()  # 创建一个空的DGLGraph对象
    g.add_nodes(ID.shape[0] + IM.shape[0])  # 添加节点到图中，节点的数量为miRNA和disease的总数
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64)  # 创建一个张量来表示节点类型，初始值为0
    node_type[: ID.shape[0]] = 1  # 将前ID.shape[0]个节点标记为1，表示它们是disease节点
    #0-374 disease
    g.ndata['type'] = node_type  # 将节点类型数据存储在图的节点特征'data'中

    d_sim = torch.zeros(g.number_of_nodes(), ID.shape[1])  # 创建一个张量来存储disease相i似性特征，初始值为0
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))  # 将从文件加载的dsease相似性特征存储在d_sim张量中
    g.ndata['d_sim'] = d_sim  # 将d_sim存储在图的节点特征'data'中



    m_sim = torch.zeros(g.number_of_nodes(), IM.shape[1])  # 创建一个张量来存储miRNA相似性特征，初始值为0
    m_sim[ID.shape[0]: ID.shape[0]+IM.shape[0], :] = torch.from_numpy(IM.astype('float32'))  # 将从文件加载的miRNA相似性特征存储在m_sim张量中
    g.ndata['m_sim'] = m_sim  # 将m_sim存储在图的节点特征'data'中




    disease_ids = list(range(1, ID.shape[0]+1))  # 创建一个包含disease节点ID的列表
    mirna_ids = list(range(1, IM.shape[0]+1))  # 创建一个包含miRNA节点ID的列表

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}  # 创建一个字典，将disease节点ID映射到索引
    mirna_ids_invmap = {id_: i for i, id_ in enumerate(mirna_ids)}  # 创建一个字典，将miRNA节点ID映射到索引

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]  # 从samples中获取disease节点的索引
    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]  # 从samples中获取miRNA节点的索引

    g.add_edges(sample_disease_vertices, sample_mirna_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})  # 在图中添加从disease到miRNA的边，带有'label'特征，特征值为samples中的第三列
    g.add_edges(sample_mirna_vertices, sample_disease_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    # e,u = eigen_decompositon(Adj)
    # e = torch.FloatTensor(e)
    # u = torch.FloatTensor(u)

    # 在图中添加从miRNA到disease的边，带有'label'特征，特征值为samples中的第三列
    g.readonly()  # 将图设置为只读，禁止修改图的结构和特征

    return g, sample_disease_vertices, sample_mirna_vertices, ID, IM, samples  # 返回构建的图以及其他相关变量
def feature_normalize(x):
    x = np.array(x)
    rowsum = x.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return x / rowsum

def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

    # std_tpr = np.std(tpr, axis=0)
    # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='LightSkyBlue', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc='lower right')
    plt.savefig(directory+'/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AP: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='BlueViolet', alpha=0.9,
             label='Mean AP: %.4f $\pm$ %.4f' % (mean_prc, prc_std))  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R curves')
    plt.legend(loc='lower left')
    plt.savefig(directory + '/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.close()
def normalize_graph(g):
    g = np.array(g)
    g = g + g.T
    g[g > 0.] = 1.0
    deg = g.sum(axis=1).reshape(-1)
    deg[deg == 0.] = 1.0
    deg = np.diag(deg ** -0.5)
    adj = np.dot(np.dot(deg, g), deg)
    L = np.eye(g.shape[0]) - adj
    return L
def eigen_decompositon(g):
    "The normalized (unit “length”) eigenvectors, "
    "such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]."
    g = normalize_graph(g)
    e, u = eigh(g)
    return e, u