import numpy as np


class KNN:
    def __init__(self, k, f):

        self.k = k
        self.f = f

    def train_model(self, train_data):

        self.data_num = train_data.data_num
        self.cat_num = train_data.cat_num
        np.random.shuffle(train_data.id)
        self.samples = np.array([train_data.feat[idx] for idx in train_data.id])
        self.raw_labels = np.array([train_data.cat[idx] for idx in train_data.id])
        self.labels = np.zeros((self.data_num, self.cat_num))
        self.labels[np.arange(self.data_num), self.raw_labels] = 1

        self.eval_model()

    def eval_model(self):

        part_size = self.data_num // self.f
        samples = list(self.samples)
        labels = list(self.labels)
        part_acc = []

        for i in range(self.f):
            s = i*part_size
            e = (i+1)*part_size
            part_samples = np.array(samples[s: e])
            part_labels = np.array(labels[s: e])
            eval_samples = np.array(samples[:s] + samples[e:])
            eval_labels = np.array(labels[:s] + labels[e:])

            part_preds = self.pred(part_samples, eval_samples, eval_labels)
            part_acc.append(np.sum(np.argmax(part_labels, 1) == part_preds)/part_size)

        print('Acc_list', part_acc)
        print('Acc_CV:\t', sum(part_acc)/self.f)

    def predict(self, test_data):

        x = np.array(test_data.feat)
        preds = self.pred(x, self.samples, self.labels)

        return preds

    def pred(self, x, samples, labels):

        logits = np.matmul(x, np.transpose(samples))
        thr = np.expand_dims(np.sort(logits, axis=1)[:, -self.k], 1)
        clipped_logits = np.clip(logits, thr, None) - thr
        scores = np.matmul(clipped_logits, labels)
        preds = np.argmax(scores, axis=1)

        return preds