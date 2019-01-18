import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP:
    def __init__(self, feat_dim, cat_num, optim, ac_fn, dr, lr, gpu,
                 max_epoch_num, bs, lamb, f, write, seed, num_seed,
                 lr_decay=False,):

        self.gpu = gpu
        self.model = MLPNet(feat_dim, cat_num, optim, ac_fn, dr, lr)
        if self.gpu > 0:
            self.model = self.model.cuda(self.gpu)
        self.optim = optim(self.model.parameters(), lr=lr)
        self.max_epoch_num = max_epoch_num
        self.init_lr = lr
        self.bs = bs
        self.lamb = lamb
        self.lr_decay = lr_decay
        self.f = f
        self.write = write
        self.seed = seed
        self.num_seed = num_seed

    def update(self, y, label, lamb, lr):

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
        loss = self.model.loss(y, label, lamb)
        loss.backward()
        self.optim.step()
        self.model.zero_grad()

        return loss

    def init_params(self):
        for layer in (self.model.fc1, self.model.fc2, self.model.fc3):
            size = layer.weight.size()
            fan_out = size[0]  # number of rows
            fan_in = size[1]  # number of columns
            variance = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, variance)
            layer.bias.data.fill_(0)

    def train_model(self, train_data, no_valid=False):
        self.data_num = train_data.data_num
        self.samples = np.array(train_data.feat)
        self.labels = np.array(train_data.cat)

        if not no_valid:
            acc_train_list = []
            acc_valid_list = []
            for i in range(self.seed, self.seed + self.num_seed):
                np.random.seed(i)
                torch.manual_seed(i)
                acc_train, acc_valid = self.eval_model()
                acc_train_list.append(acc_train)
                acc_valid_list.append(acc_valid)
            self.write(acc_train_list)
            self.write(acc_valid_list)

        train_x = train_data.feat
        train_y = train_data.cat
        self.train(train_x, train_y)
        # data = [loss_list, acc_train_list]

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

            self.train(eval_samples, eval_labels)
            part_preds = self.predict(eval_samples)
            part_train_acc.append(np.sum(eval_labels == part_preds) / (self.data_num - part_size))

            part_preds = self.predict(part_samples)
            part_valid_acc.append(np.sum(part_labels == part_preds) / part_size)

        return sum(part_train_acc) / self.f, sum(part_valid_acc) / self.f

    def train(self, train_x, train_y):

        self.init_params()
        train_size = len(train_x)
        train_ids = list(range(train_size))
        self.model.zero_grad()
        lr = self.init_lr
        loss_list = []
        acc_train_list = []

        for epoch in range(self.max_epoch_num):
            loss_sum = 0
            self.model.train()

            if self.lr_decay:
                lr = (1 - epoch / self.max_epoch_num) * self.init_lr

            np.random.shuffle(train_ids)

            feats = torch.Tensor([train_x[idx] for idx in train_ids[:self.bs]])
            cats = torch.LongTensor([train_y[idx] for idx in train_ids[:self.bs]])
            if self.gpu > -1:
                feats = feats.cuda(self.gpu)
                cats = cats.cuda(self.gpu)

            y = self.model(feats)
            loss = self.update(y, cats, self.lamb, lr)
            loss_sum += loss.detach().numpy()

            train_preds = self.predict(train_x)
            acc_train = np.sum(train_y == train_preds) / train_size

            loss_list.append(loss_sum)
            acc_train_list.append(acc_train)

    def predict(self, feat):

        self.model.eval()
        probs = self.model(torch.Tensor(feat))
        preds = torch.argmax(probs, 1).detach().numpy()

        return preds


class MLPNet(nn.Module):

    def __init__(self, feat_dim, cat_num, optim, ac_fn, dr, lr):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(feat_dim, 512, bias=True)
        self.fc2 = nn.Linear(512, 64, bias=True)
        self.fc3 = nn.Linear(64, cat_num, bias=True)
        self.dropout = nn.Dropout(dr)
        if ac_fn == 'relu':
            self.ac = F.relu
        elif ac_fn == 'tanh':
            self.ac = F.tanh
        elif ac_fn == 'sigmoid':
            self.ac = F.sigmoid
        else:
            raise NotImplementedError

        self.optim = optim(self.parameters(), lr=lr)

    def forward(self, x):

        h1 = self.dropout(self.ac(self.fc1(x)))
        h2 = self.dropout(self.ac(self.fc2(h1)))
        h3 = F.softmax(self.fc3(h2), dim=1)

        return h3

    def loss(self, y, label, lamb):

        loss = F.cross_entropy(y, label)
        loss += lamb * (torch.mean(self.fc1.weight**2) + torch.mean(self.fc2.weight**2)
                        + torch.mean(self.fc3.weight**2))

        return loss