import csv
import os


class File:
    def __init__(self, in_file, test=False):
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
        if not test:
            self.cat_num = len(set(self.cat))


class TrainFile(File):
    def __init__(self, in_file):
        super(TrainFile, self).__init__(in_file, False)


class TestFile(File):
    def __init__(self, in_file, output_path, output_file):
        super(TestFile, self).__init__(in_file, True)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.output_file = output_file

    def output(self, header, data):

        with open(self.output_file, 'w', newline='') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(header)
            for i in range(len(data)):
                file_writer.writerow(['%i' % i, str(data[i])])



