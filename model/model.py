import torch
from torch import nn as nn


# class NeuralNet(nn.Module):
#     def __init__(self, deep_col_idx, emb_col_dict, hid_layers, dropouts, out_dim):
#         super(NeuralNet, self).__init__()
#         self.emb_col_dict = emb_col_dict
#         self.deep_col_idx = deep_col_idx
#         for key, val in emb_col_dict.items():
#             setattr(self, 'dense_col_' + key, nn.Embedding(val[0], val[1]))
#         emb_layer = 0
#         for col in self.deep_col_idx.keys():
#             if col in emb_col_dict:
#                 emb_layer += emb_col_dict[col][1]
#             else:
#                 emb_layer += 1
#         self.layers = nn.Sequential()
#         dropouts = [0.0] + dropouts
#         for i in range(1, len(hid_layers)):
#             self.layers.add_module(
#                 'hidden_layer_{}'.format(i - 1),
#                 nn.Sequential(
#                     nn.Linear(hid_layers[i - 1], hid_layers[i]),
#                     nn.LeakyReLU(),
#                     nn.Dropout(dropouts[i - 1])
#                 )
#             )
#         self.layers.add_module('last_linear', nn.Linear(hid_layers[-1], out_dim))
#
#     def forward(self, x, rank):
#         emb = []
#         continuous_cols = [col for col in self.deep_col_idx.keys() if col not in self.emb_col_dict]
#
#         for col_idx, col in enumerate(rank):
#             if col in self.emb_col_dict.keys():
#                 if col not in self.deep_col_idx:
#                     raise ValueError("ERROR column name may be your deep_columns_idx dict is not math the"
#                                      "embedding_columns_dict")
#                 else:
#                     idx = self.deep_col_idx[col]
#
#
#         for col, _ in self.emb_col_dict.items():
#             if col not in self.deep_col_idx:
#                 raise ValueError("ERROR column name may be your deep_columns_idx dict is not math the"
#                                  "embedding_columns_dict")
#             else:
#                 idx = self.deep_col_idx[col]
#                 emb.append(getattr(self, 'dense_col_' + col)(x[:, idx].long()))
#
#         for col in continuous_cols:
#             idx = self.deep_col_idx[col]
#             emb.append(x[:, idx].view(-1, 1).float())
#         embedding_layers = torch.cat(emb, dim=1)
#         out = self.layers(embedding_layers)
#         return out
def linear(inp, out, dropout):
    """
    linear model module by nn.sequential
    :param inp: int, linear model input dimensio
    :param out: int, linear model output dimension
    :param dropout: float dropout probability for linear layer
    :return: tensor
    """
    return nn.Sequential(
        nn.BatchNorm1d(inp),
        nn.Linear(inp, out),
        nn.LeakyReLU(),
        nn.Dropout(dropout)
    )


class NeuralNet(nn.Module):
    def __init__(self, deep_columns_idx, embedding_columns_dict, hidden_layers, dropouts, output_dim):
        """
        init parameters
        :param deep_columns_idx: dict include column name and it's index
            e.g. {'age': 0, 'career': 1,...}
        :param embedding_columns_dict: dict include categories columns name and number of unique val and embedding dimension
            e.g. {'age':(10, 32),...}
        :param hidden_layers: number of hidden layers
        :param deep_columns_idx: dict of columns name and columns index
        :param dropouts: list of float each hidden layers dropout len(dropouts) == hidden_layers - 1
        """
        super(NeuralNet, self).__init__()
        self.embedding_columns_dict = embedding_columns_dict
        self.deep_columns_idx = deep_columns_idx
        for key, val in embedding_columns_dict.items():
            setattr(self, 'dense_col_' + key, nn.Embedding(val[0], val[1]))
        embedding_layer = 0
        for col in self.deep_columns_idx.keys():
            if col in embedding_columns_dict:
                embedding_layer += embedding_columns_dict[col][1]
            else:
                embedding_layer += 1
        self.layers = nn.Sequential()
        hidden_layers = [embedding_layer] + hidden_layers
        dropouts = [0.0] + dropouts
        for i in range(1, len(hidden_layers)):
            self.layers.add_module(
                'hidden_layer_{}'.format(i - 1),
                linear(hidden_layers[i - 1], hidden_layers[i], dropouts[i - 1])
            )
        self.layers.add_module('last_linear', nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, x):
        emb = []
        continuous_cols = [col for col in self.deep_columns_idx.keys() if col not in self.embedding_columns_dict]
        for col, _ in self.embedding_columns_dict.items():
            if col not in self.deep_columns_idx:
                raise ValueError("ERROR column name may be your deep_columns_idx dict is not math the"
                                 "embedding_columns_dict")
            else:
                idx = self.deep_columns_idx[col]
                emb.append(getattr(self, 'dense_col_' + col)(x[:, idx].long()))

        for col in continuous_cols:
            idx = self.deep_columns_idx[col]
            emb.append(x[:, idx].view(-1, 1).float())
        embedding_layers = torch.cat(emb, dim=1)
        out = self.layers(embedding_layers)
        return out

# class EFSDNN(nn.Module):
#     def __init__(self, deep_col_idx, emb_col_dict, hid_layers, dropouts, out_dim, feat_imp, num_in_feat):
#         super(EFSDNN, self).__init__()
#         self.nn = NeuralNet(deep_col_idx, emb_col_dict, hid_layers, dropouts, out_dim)
#         self.feat_imp = feat_imp
#         self.num_in_feat = num_in_feat
#         feat_dim = feat_imp.shape[1]
#         self.feat_imp_transform = nn.Linear(feat_dim, feat_dim)
#
#     def forward(self, x):
#         k = self.num_in_feat
#         feat_imp = self.feat_imp_transform(self.feat_imp).sum(0)
#         _, rank = torch.sort(feat_imp)
#         rank = rank[:k]
#         x = x[:, rank]
#         out = self.nn(x, rank)
#         return out
