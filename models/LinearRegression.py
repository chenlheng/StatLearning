from sklearn.linear_model import Ridge, LinearRegression
import numpy as np


class RidgeRegression:
    def __init__(self, lamb, f, write, seed, num_seed):
        self.model = Ridge(lamb) if lamb>0 else LinearRegression()
        self.f = f
        self.write = write
        self.seed = seed
        self.num_seed = num_seed

    def train_model(self, train_data, no_valid):
        self.data_num = train_data.data_num
        self.cat_num = train_data.cat_num
        self.samples = np.array(train_data.feat)
        raw_labels = np.array(train_data.cat)
        self.labels = np.zeros((self.data_num, self.cat_num))
        self.labels[np.arange(self.data_num), raw_labels] = 1

        if not no_valid:
            acc_train_list = []
            acc_valid_list = []
            for i in range(self.seed, self.seed + self.num_seed):
                np.random.seed(i)
                acc_train, acc_valid = self.eval_model()
                acc_train_list.append(acc_train)
                acc_valid_list.append(acc_valid)
            self.write(acc_train_list)
            self.write(acc_valid_list)

        self.model.fit(self.samples, self.labels)

    def eval_model(self):
        part_size = self.data_num // self.f
        idx = list(range(self.data_num))
        np.random.shuffle(idx)
        samples = [list(self.samples)[i] for i in idx]
        labels = [list(self.labels)[i] for i in idx]
        part_train_acc = []
        part_valid_acc = []

        for i in range(self.f):
            s = i * part_size
            e = (i + 1) * part_size
            part_samples = np.array(samples[s: e])
            part_labels = np.array(labels[s: e])
            eval_samples = np.array(samples[:s] + samples[e:])
            eval_labels = np.array(labels[:s] + labels[e:])

            self.model.fit(eval_samples, eval_labels)
            part_preds = np.argmax(self.model.predict(eval_samples), 1)
            part_train_acc.append(np.sum(np.argmax(eval_labels, 1) == part_preds) / (self.data_num - part_size))
            part_preds = np.argmax(self.model.predict(part_samples), 1)
            part_valid_acc.append(np.sum(np.argmax(part_labels, 1) == part_preds) / part_size)

        return sum(part_train_acc) / self.f, sum(part_valid_acc) / self.f

    def predict(self, feat):

        preds = self.model.predict(np.array(feat))

        return np.argmax(preds, 1)
