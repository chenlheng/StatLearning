import xgboost as xgb
import sklearn
import numpy as np


class XgBoost:
    def __init__(self, max_depth, eta, nthread, lamb, sub_sample):
        self.params = {
            'bst:max_depth': max_depth,
            'bst:eta': eta,  #
            'bst:lambda': lamb,  # l2_coef
            'bst:sub_sample': sub_sample,
            'silent': 1,
            'objective': 'multi:softmax',
            'nthread': nthread,
            'eval_metric': 'merror',
            'num_class': 12
        }

        self.param_list = self.params.items()

    def train_model(self, train_data, max_epoch_num, no_valid):

        if not no_valid:
            valid_size = train_data.data_num // 5
            train_size = train_data.data_num - valid_size
            train_x = train_data.feat[:-valid_size]
            train_y = train_data.cat[:-valid_size]
            dtrain = xgb.DMatrix(train_x, label=train_y)
            valid_x = train_data.feat[-valid_size:]
            valid_y = train_data.cat[-valid_size:]
            dvalid = xgb.DMatrix(valid_x, label=valid_y)
            eval_list = [(dtrain, 'train'), (dvalid, 'eval')]
        else:
            train_size = train_data.data_num
            train_x = train_data.feat[:train_size]
            train_y = train_data.cat[:train_size]
            dtrain = xgb.DMatrix(train_x, label=train_y)
            eval_list = [(dtrain, 'train')]

        # train_ids = list(range(train_size))
        self.bst = xgb.train(self.params, dtrain, max_epoch_num, eval_list)

        train_preds = self.predict(train_x)
        acc_train = sklearn.metrics.accuracy_score(train_y, train_preds)

        if not no_valid:
            valid_preds = self.predict(valid_x)
            acc_valid = sklearn.metrics.accuracy_score(valid_y, valid_preds)
            print(acc_train, acc_valid)
        else:
            print(acc_train)

    def predict(self, feat):

        preds = self.bst.predict(xgb.DMatrix(feat))

        return preds

