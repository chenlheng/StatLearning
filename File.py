import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
import numpy as np


class File:
    def __init__(self, in_file, test=False, shuffle=True):
        print('reading %s' % in_file)

        self.id = []
        self.feat = []
        if not test:
            self.cat = []
        with open(in_file, newline='') as csv_file:
            file_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
            # [<int>'id', <float>'feature_%i' % (0-4095), <int>'catagories']
            i = 0
            for row in file_reader:
                if i > 0:
                    row_len = len(row)
                    self.id.append(int(row[0]))
                    if not test:
                        self.feat.append(list(map(float, row[1:row_len-1])))
                        self.cat.append(int(row[-1]))
                    else:
                        self.feat.append(list(map(float, row[1:row_len])))
                i += 1
        self.feat_dim = len(self.feat[0])
        self.data_num = len(self.id)

        if shuffle:
            np.random.shuffle(self.id)
            self.feat = [self.feat[idx] for idx in self.id]
            if not test:
                self.cat = [self.cat[idx] for idx in self.id]

        if not test:
            self.cat_num = len(set(self.cat))


class TrainFile(File):
    def __init__(self, in_file, normalize=True):
        super(TrainFile, self).__init__(in_file, test=False, shuffle=True)
        if normalize:
            self.raw_feat = self.feat[:]
            normalized_train_x = self.raw_feat - np.mean(self.raw_feat, 0)
            self.feat = normalized_train_x / np.var(normalized_train_x)


class TestFile(File):
    def __init__(self, in_file, train_data, output_path, output_file, normalize=True):
        super(TestFile, self).__init__(in_file, test=True, shuffle=False)

        # Normalize test_feat according to train_feat
        if normalize:
            self.normalize(train_data)

        self.output_path = output_path
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.output_file = output_file
        self.color_list = ['b', 'g', 'r']
        self.f = open(self.output_path+'results.txt', 'w')

    def write(self, message):
        print(message)
        self.f.write(str(message) + '\n')

    def output(self, header, data):

        with open(self.output_file, 'w', newline='') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(header)
            for i in range(len(data)):
                file_writer.writerow(['%i' % i, str(int(data[i]))])

    def draw(self, data):
        idx = np.arange(len(data[0]))
        for i in range(len(data)):
            color = self.color_list[i]
            curve = np.array(data[i])
            if curve[0] < np.mean(curve):
                fn = np.argmax
            else:
                fn = np.argmin
            plt.plot(idx, curve, color=color)
            x = fn(curve)
            if not x == idx[-1]:
                plt.scatter([idx[-1]], [curve[-1]], color=color)
                plt.annotate('%.4f' % (curve[-1]), [idx[-1], curve[-1]])
            plt.scatter([x], [curve[x]], color=color)
            plt.annotate('%i-%.4f' % (x, curve[x]), [x, curve[x]])

        plt.savefig(self.output_path + 'training.jpg')

    def normalize(self, train_data):
        raw_feat = train_data.raw_feat
        normalized_train_x = raw_feat - np.mean(raw_feat, 0)
        self.feat = normalized_train_x / np.var(normalized_train_x)