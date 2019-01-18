from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
import numpy as np


class NaiveBayes:
    def __init__(self, model_type, f, write, seed, num_seed):
        self.model_type = model_type
        if model_type == 'multinomial':
            self.model = MultinomialNB()
        elif model_type == 'gaussian':
            self.model = GaussianNB()
        elif model_type == 'bernoulli':
            self.model = BernoulliNB()
        else:
            raise NotImplementedError

        self.f = f
        self.write = write
        self.seed = seed
        self.num_seed = num_seed

    def train_model(self, train_data, no_valid):
        self.data_num = train_data.data_num
        self.samples = np.array(train_data.feat)
        if self.model_type == 'multinomial':
            self.samples -= np.min(self.samples)
        self.labels = np.array(train_data.cat)

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
            part_preds = self.model.predict(eval_samples)
            part_train_acc.append(np.sum(eval_labels == part_preds) / (self.data_num-part_size))
            part_preds = self.model.predict(part_samples)
            part_valid_acc.append(np.sum(part_labels == part_preds) / part_size)

        return sum(part_train_acc) / self.f, sum(part_valid_acc) / self.f

    def predict(self, feat):

        x = np.array(feat)
        if self.model_type == 'multinomial':
            x -= np.min(x)
        preds = self.model.predict(x)

        return preds
