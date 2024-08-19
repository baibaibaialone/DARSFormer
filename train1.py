import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn import metrics
import scipy.sparse as sp
from scipy.sparse import csc_matrix, coo_matrix
import math
from cul_loss import CustomLossWithRegularization
import logging
import matplotlib.pyplot as plt


def normalize_adjacency_matrix(adj):
    # 添加自连接（self-loops）
    adj = adj + torch.eye(adj.size(0))

    # 计算度矩阵D
    degrees = torch.sum(adj, dim=1)

    # 度矩阵D的逆开平方
    D_inv_sqrt = torch.diag(torch.pow(degrees, -0.5))

    # 归一化邻接矩阵： D^(-0.5) * A * D^(-0.5)
    adj_normalized = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
    adj = adj_normalized

    return adj
def eigenvalue_encoding(e, constant, hidden_dim=64):
    ee = e * constant
    div = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000) / hidden_dim))
    pe = ee.unsqueeze(1) * div
    eeig = torch.cat((e.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)
    return eeig
# 配置日志输出到文件
logging.basicConfig(filename='output.log', level=logging.INFO, format='%(message)s')

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx
def process_adj(adj):
    adj.setdiag(1)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj
def edge_index_to_sparse_mx(edge_index, num_nodes):
    edge_weight = np.array([1] * len(edge_index[0]))
    adj = csc_matrix((edge_weight, (edge_index[0], edge_index[1])),
                     shape=(num_nodes, num_nodes)).tolil()
    return adj
from utils import load_data, build_graph, weight_reset, eigen_decompositon
from model2 import  Specformer
def Train(directory, epochs, n_classes, in_size, out_dim, dropout, slope, lr, wd, random_seed, cuda):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        if not cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed(random_seed)


    context = torch.device('cpu')

    g, disease_vertices, mirna_vertices, ID, IM, samples = build_graph(directory, random_seed)

    samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])
    #dgl
    adj_max = g.adjacency_matrix().to_dense()
    adj_max = torch.FloatTensor(adj_max)
    # e, u = eigen_decompositon(adj_m)
    # np.savetxt(r'C:\Users\Lenovo\Desktop\代码调试\备份\bx-7 +10 - los - test - syx\eigenvalues.csv', e, delimiter=',')
    # # 保存eigenvectors
    # np.savetxt(r'C:\Users\Lenovo\Desktop\代码调试\备份\bx-7 +10 - los - test - syx\eigenvectors.csv', u, delimiter=',')

    # e=torch.Tensor(e)
    # u =torch.Tensor(u)
    # d=64
    #
    # indices = torch.randperm(878)[:50]
    # e1 = e[indices]
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    #
    # constants = [1, 10, 100,]
    #
    # for i, constant in enumerate(constants):
    #     eeig = eigenvalue_encoding(e1, constant)
    #     # 仅使用前20维进行可视化，并设置横坐标范围，确保图像的起始点位于左下角
    #     ax[i].imshow(eeig[:, :20].numpy(), aspect='auto', cmap='viridis', extent=[0, 60, 0, 50], origin='lower')
    #     ax[i].set_title(f'λ = {constant}')
    #     ax[i].set_xlabel('Dimension')
    #     ax[i].set_ylabel('Eigenvalue')
    #     ax[i].set_xticks([0, 20, 40, 60])
    #
    # plt.tight_layout()
    # plt.show()

    #draw





    # e = torch.FloatTensor(e)
    # u = torch.FloatTensor(u)

    ##add
    list_f = list(zip(disease_vertices, mirna_vertices))
    transposed_list = [[row[i] for row in list_f] for i in range(2)]
    transposed_list = torch.FloatTensor(transposed_list)
    adj_q = edge_index_to_sparse_mx(transposed_list, 1162)
    adj_10 = process_adj(adj_q)











    # adj_m = Adj
    # adj_m = torch.Tensor(adj_m)
    # print(np.array_equal(adj_max,adj_m))
   # print('## vertices:', g.number_of_nodes())
    print('## edges:', g.number_of_edges())
    #????????????????
    print('## disease nodes:', torch.sum(g.ndata['type'] == 1).numpy())#374
    print('## mirna nodes: ', torch.sum(g.ndata['type'] == 0).numpy()) #788


    g.to(context)



    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []
    prc_result = []

    fprs = []
    tprs = []
    precisions = []
    recalls = []
    patience = 150

    i = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    for train_idx, test_idx in kf.split(samples[:, 2]):     # 返回训练集和测试集的索引train：test 4:1
        best_val_auc = 0.0

        i += 1
        print('Training for Fold', i)

        samples_df['train'] = 0
        samples_df['train'].iloc[train_idx] = 1 # 多加一列，将训练集记为1

        train_tensor = torch.from_numpy(samples_df['train'].values.astype('int64'))

        edge_data = {'train': train_tensor}

        g.edges[disease_vertices, mirna_vertices].data.update(edge_data)        # 正向反向加边，更新边上的数据
        g.edges[mirna_vertices, disease_vertices].data.update(edge_data)
        #???????

        train_eid = g.filter_edges(lambda edges: edges.data['train'])       # 过滤出被记为train的边
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)       # 从异构图中创建子图，train集的子图
        # g_train.copy_from_parent()
        label_train = g_train.edata['label'].unsqueeze(1)
        src_train, dst_train = g_train.all_edges()          # 训练集的边

        test_eid = g.filter_edges(lambda edges: edges.data['train'] == 0)   # 原图中选出标记为0的记为测试集
        src_test, dst_test = g.find_edges(test_eid)
        label_test = g.edges[test_eid].data['label'].unsqueeze(1)       # 测试集的边
        print('## Training edges:', len(train_eid))
        print('## Testing edges:', len(test_eid))

        adj_m = g_train.adjacency_matrix().to_dense()
        adj_m = torch.FloatTensor(adj_m)
        adj_m=normalize_adjacency_matrix(adj_m)

        e, u = eigen_decompositon(adj_m)
        e = torch.FloatTensor(e)
        u = torch.FloatTensor(u)

        model = Specformer(
            adj = adj_m,
            adjmax = adj_max,
                           transposed_list = transposed_list,

                           e = e,
                           u = u,
            G=g_train,
                        hid_dim=in_size,
                        n_class=n_classes,
                        S=3,
                        K=4,
                        batchnorm=False,
                        num_diseases=ID.shape[0],
                        num_mirnas=IM.shape[0],
                        d_sim_dim=ID.shape[1],
                        m_sim_dim=IM.shape[1],
                        out_dim=out_dim,
                        dropout=dropout,
                        slope=slope,
                        node_dropout=0,
                        input_droprate=0,
                        hidden_droprate=0,

                        nclass=64,
                        nfeat=64,
                           nlayer=2 ,
            # 94.07,94.17,94.18,94.18,85.41,86.86,94.18,86.52,
                           hidden_dim=64,
                           nheads=2,
                           tran_dropout=0,
                           feat_dropout=0,
                           prop_dropout=0,
                           norm='none'

                      )

        model.apply(weight_reset)
        model.to(context)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        lambda_reg = 0.001  # 调整此值以控制正则化的强度
        cul_loss = CustomLossWithRegularization(lambda_reg)

        for epoch in range(epochs):
            start = time.time()

            model.train()
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                score_train = model(g_train, src_train, dst_train, False)  # train集子图进入model训练
                loss_train = cul_loss(score_train, label_train, model)
                # loss_train += 0.001 * model.ortho_loss()

                # optimizer.zero_grad()   # 梯度置零
                loss_train.backward()   # 反向传播
                optimizer.step()

            model.eval()
            with torch.no_grad():   # with torch.no_grad()或者@torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
                score_val = model(g, src_test, dst_test, True)    # 注意在整个图g中训练测试集
                loss_val = cul_loss(score_val, label_test,model)

            score_train_cpu = np.squeeze(score_train.cpu().detach().numpy())        # 在深度学习训练后，需要计算每个epoch得到的模型的训练效果的时候，
            score_val_cpu = np.squeeze(score_val.cpu().detach().numpy())            # 一般会用到detach() item() cpu() numpy()等函数。
            label_train_cpu = np.squeeze(label_train.cpu().detach().numpy())
            label_val_cpu = np.squeeze(label_test.cpu().detach().numpy())

            train_auc = metrics.roc_auc_score(label_train_cpu, score_train_cpu)
            val_auc = metrics.roc_auc_score(label_val_cpu, score_val_cpu)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1

                # 检查是否应该停止
            if patience_counter >= patience:
                break

            pred_val = [0 if j < 0.5 else 1 for j in score_val_cpu]
            acc_val = metrics.accuracy_score(label_val_cpu, pred_val)
            pre_val = metrics.precision_score(label_val_cpu, pred_val)
            recall_val = metrics.recall_score(label_val_cpu, pred_val)
            f1_val = metrics.f1_score(label_val_cpu, pred_val)

            end = time.time()
            print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),
                  'Val Loss: %.4f' % loss_val.cpu().detach().numpy(),
                  'Acc: %.4f' % acc_val, 'Pre: %.4f' % pre_val, 'Recall: %.4f' % recall_val, 'F1: %.4f' % f1_val,
                  'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc)

            # if (epoch + 1) % 10 == 0:
            #     print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),
            #           'Val Loss: %.4f' % loss_val.cpu().detach().numpy(),
            #           'Acc: %.4f' % acc_val, 'Pre: %.4f' % pre_val, 'Recall: %.4f' % recall_val, 'F1: %.4f' % f1_val,
            #           'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc)
            #     message = f"Fold {i}: {val_auc:.4f}"
            #     logging.info(message)
        model.load_state_dict(best_model_state)
        model.eval()


        with torch.no_grad():
            model.load_state_dict(best_model_state)
            score_test = model(g, src_test, dst_test, True)   # 测试分数和验证分数相同？？？


        score_test_cpu = np.squeeze(score_test.cpu().detach().numpy())  # np.squeeze删除指定的维度
        label_test_cpu = np.squeeze(label_test.cpu().detach().numpy())

        fpr, tpr, thresholds = metrics.roc_curve(label_test_cpu, score_test_cpu)
        precision, recall, _ = metrics.precision_recall_curve(label_test_cpu, score_test_cpu)
        test_auc = metrics.auc(fpr, tpr)
        test_prc = metrics.auc(recall, precision)

        pred_test = [0 if j < 0.5 else 1 for j in score_test_cpu]
        acc_test = metrics.accuracy_score(label_test_cpu, pred_test)
        pre_test = metrics.precision_score(label_test_cpu, pred_test)
        recall_test = metrics.recall_score(label_test_cpu, pred_test)
        f1_test = metrics.f1_score(label_test_cpu, pred_test)
        print(f"Fold completed. Best validation AUC: {best_val_auc}")


        print('Fold: ', i, 'Test acc: %.4f' % acc_test, 'Test Pre: %.4f' % pre_test,
              'Test Recall: %.4f' % recall_test, 'Test F1: %.4f' % f1_test, 'Test PRC: %.4f' % test_prc,
              'Test AUC: %.4f' % best_val_auc)

        auc_result.append(best_val_auc)
        acc_result.append(acc_test)
        pre_result.append(pre_test)
        recall_result.append(recall_test)
        f1_result.append(f1_test)
        prc_result.append(test_prc)

        fprs.append(fpr)
        tprs.append(tpr)
        precisions.append(precision)
        recalls.append(recall)

    print('## Training Finished !')
    print('-----------------------------------------------------------------------------------------------')
    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
          'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))

    return fprs, tprs, auc_result, precisions, recalls, prc_result
