import numpy as np


class EM:
    def __init__(self):
        pass

    def e_step(self, x):
        pass

    def m_step(self):
        pass

    def train_model(self, train_data, max_step):
        x = np.array(train_data.feat)  # N x feat_dim
        y = np.expand_dims(np.array(train_data.cat), 1)  # N x 1
        last_acc = 0
        count = 0
        step = 0

        while count < 5 and step < max_step:


            if last_acc == acc:
                count += 1
            else:
                count = 0

            step += 1

    def predict(self):
        pass
