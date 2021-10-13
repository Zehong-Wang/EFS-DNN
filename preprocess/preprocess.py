import copy
import sys

import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


parser = argparse.ArgumentParser('Preprocess dataset')
parser.add_argument('-d', '--data', type=str, help='Dataset name', default='kdd99')
parser.add_argument('--classes', type=int, choices=[2, 5], help='The attach class', default=2)

try:
    args = parser.parse_args()
except BaseException:
    parser.print_help()
    sys.exit(0)

df = pd.read_csv(f'./data/{args.data}_raw.csv')

features = df.iloc[:, :-1]
target = df.iloc[:, -1]

numerical_features = [x for x in features.columns if features[x].dtype == np.int64 or features[x].dtype == np.float64]
categorical_features = [x for x in features.columns if features[x].dtype == object]

lbe = LabelEncoder()
for feat in categorical_features:
    df[feat] = lbe.fit_transform(df[feat])

mms = MinMaxScaler()
df[numerical_features] = mms.fit_transform(df[numerical_features])

if args.classes == 2:
    mask = target == 'normal'
    target[mask] = 0
    target[~mask] = 1
elif args.classes == 5:
    attack_type = pd.read_csv('./data/attack_type.csv')
    attack2type = dict()
    # type_set = set(attack_type['type'])
    for attack, type in attack_type.values:
        attack2type[attack] = type
    # type2idx = {type: idx for idx, type in enumerate(type_set)}
    type2idx = {'normal': 0, 'dos': 1, 'probe': 2, 'r2l': 3, 'u2r': 4}
    df.iloc[:, -1] = target.apply(lambda x: type2idx[attack2type[x]])

df.to_csv(f'./data/{args.data}_{args.classes}_pre.csv', index=False)