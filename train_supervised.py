import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, KFold
import time
from pycm import ConfusionMatrix
from model.model import NeuralNet
from utils.args import get_args
from utils.utils import get_feature_importance

args = get_args()
DATA = args.data
GPU = args.gpu
SEED = args.seed
N_LGB = args.n_lgb
LR = args.lr
CLASSES = args.classes
R_SAMPLE = args.r_sample
EPOCH = args.n_epoch
BS = args.bs
THRESHOLD = args.threshold
# NUM_IN_FEAT = args.num_in_feat


class IDSDataset(Dataset):
    def __init__(self, features, target, transform=None):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        return self.features[item], self.target[item]


def run():
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)

    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() and GPU >= 0 else 'cpu'
    print(f'Model is trained in {device_string}')
    device = torch.device(device_string)

    df = pd.read_csv(f'./data/{DATA}_{CLASSES}_pre.csv')

    feat_select_start = time.time()
    feats = df.iloc[:, :-1]
    target = df.iloc[:, -1]
    # print('categorical features are ', cate_feats)
    # print('numerical features are ', num_feats)

    feat_imp = get_feature_importance(feats.values, target.values, N_LGB, R_SAMPLE)
    feat_imp = feat_imp.sum(0)
    rank = np.argsort(-feat_imp)
    feat_imp = -(np.sort(-feat_imp) / feat_imp.sum())
    feat_imp = feat_imp.cumsum()
    num_in_feat = np.where(feat_imp > THRESHOLD)[0] + 1

    feats = feats.iloc[:, rank[:num_in_feat]]
    # feat_imp = torch.tensor(feat_imp, dtype=torch.float, device=device)
    feat_select_time = time.time() - feat_select_start

    cate_feats = [x for x in feats.columns if feats[x].dtype == np.int64]
    num_feats = [x for x in feats.columns if feats[x].dtype == np.float64]
    feat_dict = {feat: idx for idx, feat in enumerate(feats.columns)}

    embedding_dim = 5
    embedding_feat = {feat: (feats[feat].value_counts().count(), embedding_dim) for feat in cate_feats}

    efsdnn = NeuralNet(feat_dict, embedding_feat, [512, 512, 512], [0, 0, 0], CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(efsdnn.parameters(), lr=LR)

    for i in range(EPOCH):
        x_train, x_test, y_train, y_test = train_test_split(feats.values, target.values, train_size=0.8, shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.875, shuffle=True)
        train_dataset = IDSDataset(x_train, y_train)
        val_dataset = IDSDataset(x_val, y_val)
        test_dataset = IDSDataset(x_test, y_test)
        train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BS, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=BS, shuffle=True)

        train_start = time.time()
        loss = train(train_dataloader, efsdnn, loss_fn, optim, device)
        train_time = time.time() - train_start + feat_select_time

        print('Training time: {:.8f}'.format(train_time))

        print('---------------Validation-----------------')
        val_acc, val_f1, val_fpr, val_tpr, val_auc, val_inf_time = eval(val_dataloader, efsdnn, CLASSES, device)
        print_results(val_acc, val_f1, val_fpr, val_tpr, val_auc, val_inf_time)

        print('---------------Test-----------------')
        test_acc, test_f1, test_fpr, test_tpr, test_auc, test_inf_time = eval(test_dataloader, efsdnn, CLASSES, device)
        print_results(test_acc, test_f1, test_fpr, test_tpr, test_auc, test_inf_time)


def print_results(acc, f1, fpr, tpr, auc, inf_time):
    classes = len(list(acc.keys())) - 1
    if classes == 2:
        print('Accuracy: {:.4f}, FPR: {:.4f}, TPR: {:.4f}, AUC: {:.4f}, F1: {:.4f}'.format(
            acc[2], fpr[2], tpr[2], auc[2], f1[2]
        ))
        print('Inference time: {:.8f}'.format(inf_time))
    elif classes == 5:
        # type2idx = {'normal': 0, 'dos': 1, 'probe': 2, 'r2l': 3, 'u2r': 4}
        idx2type = {0: 'Normal', 1: 'DOS', 2: 'Probe', 3: 'R2L', 4: 'U2R', 5: 'Overall'}
        # print('Accuracy')
        for idx, type in idx2type.items():
            print('Type: {}, Accuracy: {:.4f}, FPR: {:.4f}, TPR: {:.4f}, AUC: {:.4f}, F1: {:.4f}'.format(
                type, acc[idx], fpr[idx], tpr[idx], auc[idx], f1[idx]
            ))
        print('Inference time: {:.8f}'.format(inf_time))


def train(dataloader, model, loss_fn, optim, device):
    model.train()
    m_loss = []

    for X, y in dataloader:
        optim.zero_grad()

        X, y = X.to(device), y.to(device)
        pred_y = model(X)
        loss = loss_fn(pred_y, y)

        loss.backward()
        optim.step()
        m_loss.append(loss.item())

    return np.mean(m_loss)


def eval(dataloader, model, classes, device):
    model.eval()
    m_acc, m_tpr, m_fpr, m_auc, m_f1 = [], [], [], [], []
    d_acc, d_tpr, d_fpr, d_auc, d_f1 = {}, {}, {}, {}, {}
    m_inf_time = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)

            eval_start = time.time()
            y_pred = torch.softmax(model(X).detach().cpu(), dim=1).numpy()
            m_inf_time.append(time.time() - eval_start)

            y_l = y_pred.argmax(1)
            cm = ConfusionMatrix(actual_vector=y.numpy(), predict_vector=y_l)
            m_acc.append(cm.ACC)
            m_tpr.append(cm.TPR)
            m_fpr.append(cm.FPR)
            m_f1.append(cm.F1)
            m_auc.append(cm.AUC)

    for k in range(classes):
        d_acc[k] = np.mean([d[k] if d[k] != 'None' else 0 for d in m_acc if k in set(d.keys())])
        d_f1[k] = np.mean([d[k] if d[k] != 'None' else 0 for d in m_f1 if k in set(d.keys())])
        d_fpr[k] = np.mean([d[k] if d[k] != 'None' else 0 for d in m_fpr if k in set(d.keys())])
        d_tpr[k] = np.mean([d[k] if d[k] != 'None' else 0 for d in m_tpr if k in set(d.keys())])
        d_auc[k] = np.mean([d[k] if d[k] != 'None' else 0 for d in m_auc if k in set(d.keys())])
    d_acc[classes] = np.mean(list(d_acc.values()))
    d_f1[classes] = np.mean(list(d_f1.values()))
    d_fpr[classes] = np.mean(list(d_fpr.values()))
    d_tpr[classes] = np.mean(list(d_tpr.values()))
    d_auc[classes] = np.mean(list(d_auc.values()))

    return d_acc, d_f1, d_fpr, d_tpr, d_auc, np.mean(m_inf_time)


if __name__ == '__main__':
    run()
