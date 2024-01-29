from sklearn import metrics

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

plt.figure(0).clf() # plt.close()将完全关闭图形窗口，其中plt.clf()将清除图形-您仍然可以在其上绘制另一个绘图。
# label = ([1, 1, 1, 1, 1, 1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])  # 非二进制需要pos_label
# pred = ([0.8, 0.8, 0.8, 0.8, 0.8, 0.8,0.8,0.8,0.8,0.8,0.8,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.5])
# fpr, tpr, thersholds = metrics.roc_curve(label, pred)
# # pred = np.random.rand(1000)
# #
# # label = np.random.randint(2, size=1000)
# #
# # fpr, tpr, thresh = metrics.roc_curve(label, pred)
#
# auc = metrics.roc_auc_score(label, pred)
#
# plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))

# pred = np.random.rand(1000)
#
# label = np.random.randint(2, size=1000)
#
# fpr, tpr, thresh = metrics.roc_curve(label, pred)
label = ([1, 1, 1, 1, 1, 1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1, 1, 1, 1, 1, 1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1, 1, 1, 1, 1, 1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])  # 非二进制需要pos_label
pred = ([0.11, 1, 0.12, 1, 0.13, 0.14,0.15,0.16,0.17,0.18,0.8,0.19,0.111,0.1112,0.112,0.1,0.1,0.1,0.1,0.1,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0.8, 0.8, 0.8, 0.8, 0.8, 0.8,0.8,0.8,0.8,0.8,0.8,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0.8, 0.8, 0.8, 0.8, 0.8, 0.8,0.8,0.8,0.8,0.8,0.8,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
fpr, tpr, thersholds = metrics.roc_curve(label, pred)
auc = metrics.roc_auc_score(label, pred)

plt.plot(fpr, tpr, label="data 2, auc=" + str(auc))

plt.legend(loc=0) # 说明所在位置
plt.show()

# mean_fpr = np.linspace(0, 1, 200)
# tprs = []
#
# for i in range(len(fpr)):
#     tprs.append(interp(mean_fpr, fpr[i], tpr[i]))
#     tprs[-1][0] = 0.0
#     plt.plot(fpr[i], tpr[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))
#
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# # mean_auc = metrics.auc(mean_fpr, mean_tpr)
# mean_auc = np.mean(auc)
# auc_std = np.std(auc)
# plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))
#
# plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

# std_tpr = np.std(tpr, axis=0)
# tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
# tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
# plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='LightSkyBlue', alpha=0.3, label='$\pm$ 1 std.dev.')

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curves')
# plt.legend(loc='lower right')
# plt.show()
# plt.savefig(directory+'/%s.jpg' % name, dpi=1200, bbox_inches='tight')
# plt.close()
