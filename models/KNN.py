import numpy as np


class KNN:
    def __init__(self, k, f, write, seed, num_seed):

        self.k = k
        self.f = f
        self.write = write
        self.seed = seed
        self.num_seed = num_seed

    def train_model(self, train_data, no_valid):

        self.data_num = train_data.data_num
        self.cat_num = train_data.cat_num
        self.samples = np.array(train_data.feat)
        self.raw_labels = np.array(train_data.cat)
        self.labels = np.zeros((self.data_num, self.cat_num))
        self.labels[np.arange(self.data_num), self.raw_labels] = 1

        if not no_valid:
            acc_train_list = []
            acc_valid_list = []
            for i in range(self.seed, self.seed+self.num_seed):
                np.random.seed(i)
                acc_train, acc_valid = self.eval_model()
                acc_train_list.append(acc_train)
                acc_valid_list.append(acc_valid)
            self.write(acc_train_list)
            self.write(acc_valid_list)

    def eval_model(self):

        part_size = self.data_num // self.f
        idx = list(range(self.data_num))
        np.random.shuffle(idx)
        samples = [list(self.samples)[i] for i in idx]
        labels = [list(self.labels)[i] for i in idx]
        part_train_acc = []
        part_valid_acc = []

        for i in range(self.f):
            s = i*part_size
            e = (i+1)*part_size
            part_samples = np.array(samples[s: e])
            part_labels = np.array(labels[s: e])
            eval_samples = np.array(samples[:s] + samples[e:])
            eval_labels = np.array(labels[:s] + labels[e:])

            part_preds = self.pred(eval_samples, eval_samples, eval_labels)
            part_train_acc.append(np.sum(np.argmax(eval_labels, 1) == part_preds) / (self.data_num-part_size))

            part_preds = self.pred(part_samples, eval_samples, eval_labels)
            part_valid_acc.append(np.sum(np.argmax(part_labels, 1) == part_preds)/part_size)

        return sum(part_train_acc)/self.f, sum(part_valid_acc)/self.f

    def predict(self, feat):

        x = np.array(feat)
        preds = self.pred(x, self.samples, self.labels)

        return preds

    def pred(self, x, samples, labels):

        logits = np.matmul(x, np.transpose(samples))
        thr = np.expand_dims(np.sort(logits, axis=1)[:, -self.k], 1)
        clipped_logits = np.clip(logits, thr, None) - thr
        scores = np.matmul(clipped_logits, labels)
        preds = np.argmax(scores, axis=1)

        return preds