import warnings
import pickle
from train1 import Train
from utils import plot_auc_curves, plot_prc_curves
def save_to_file(variable, filename):
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    fprs, tprs, auc, precisions, recalls, prc = Train(directory='data',
                                                      epochs=1000,                                                      n_classes=64,
                                                      in_size=64,
                                                      out_dim=64,
                                                      dropout=0,
                                                      slope=0.2,
                                                      lr=0.001,
                                                      wd=5e-3,
                                                      random_seed=1235,
                                                      cuda=True)
    save_to_file(fprs, "fprs.pkl")
    save_to_file(tprs, "tprs.pkl")
    save_to_file(auc, "auc.pkl")
    save_to_file(precisions, "precisions.pkl")
    save_to_file(recalls, "recalls.pkl")
    save_to_file(prc, "prc.pkl")


    plot_auc_curves(fprs, tprs, auc, directory='roc_result', name='test_auc')
    plot_prc_curves(precisions, recalls, prc, directory='roc_result', name='test_prc')