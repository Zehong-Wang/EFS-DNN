import lightgbm as lgb
import numpy as np


def get_feature_importance(features, target, num_lgb, r_sample):
    feature_importance = np.zeros([num_lgb, features.shape[1]])
    for i in range(num_lgb):
        clf = lgb.sklearn.LGBMClassifier(subsample=r_sample, subsample_freq=1, random_state=i)
        clf.fit(features, target)
        feature_importance[i] = clf.feature_importances_

    return feature_importance
