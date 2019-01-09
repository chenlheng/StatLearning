import torch
import torch.nn as nn
import numpy as np


class LinearSVM(nn.Module):
    def __init__(self, input_dim, optim, lr):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.optim = optim(self.parameters(), lr=lr)

    def update(self, y, label, lamb, lr):

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
        loss = torch.mean(torch.clamp(1 - y * label, min=0))  # hinge loss
        loss += lamb * torch.mean(self.fc.weight ** 2)  # l2 penalty
        loss.backward()
        self.optim.step()

        return loss

    def forward(self, x):
        h = self.fc(x)
        return h

    def train_model(self, train_data, max_epoch_num, init_lr, bs, lamb, lr_decay=False):

        lr = init_lr

        for epoch in range(max_epoch_num):
            np.random.shuffle(train_data.id)
            start_no = 0
            loss_sum = 0

            if lr_decay:
                lr = (1 - epoch / max_epoch_num) * init_lr

            while start_no + bs <= train_data.data_num:
                bs_id = train_data.id[start_no: start_no + bs]
                feats = torch.Tensor([train_data.feat[id] for id in bs_id])
                cats = torch.Tensor([train_data.cat[id] for id in bs_id])

                self.zero_grad()
                y = self.forward(feats)
                loss = self.update(y, cats, lamb, lr)
                loss_sum += loss.detach().numpy()

                start_no += bs

            print(epoch, loss_sum)

    def predict(self, test_data):

        y = self.forward(torch.Tensor(test_data.feat)).detach().numpy()[:, 0]
        preds = y

        return preds