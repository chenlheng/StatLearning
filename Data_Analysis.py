import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from File import TrainFile, TestFile
import numpy as np
import argparse
from sklearn.manifold import TSNE


def parse_arg():

    parser = argparse.ArgumentParser()

    parser.add_argument('-n_samples', type=int, default=200)
    parser.add_argument('-input_path', type=str, default='/home/lhchen/nas/statlearning/data/')
    parser.add_argument('-train_file', type=str, default='train.csv')
    parser.add_argument('-test_file', type=str, default='test.csv')
    parser.add_argument('-output_path', type=str, default='/home/lhchen/nas/res/stat/')
    parser.add_argument('-output_file', type=str, default='res.csv')
    parser.add_argument('-seed', type=int, default=0)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_arg()
    np.random.seed(args.seed)
    n = args.n_samples

    train_file = args.input_path + args.train_file
    train_data = TrainFile(train_file, False)

    test_file = args.input_path + args.test_file
    output_path = args.output_path + 'test/'
    output_file = args.output_path + 'test/' + args.output_file
    test_data = TestFile(test_file, train_data.feat, output_path, output_file, False)

    train_x = np.array(train_data.feat)
    test_x = np.array(test_data.feat)
    np.random.shuffle(train_x)
    np.random.shuffle(test_x)

    tsne = TSNE(n_components=2, random_state=0)
    trans_train_x = tsne.fit_transform(train_x[:n])
    for i in range(n):
        plt.scatter(trans_train_x[i, 0], trans_train_x[i, 1], c='g')
    trans_test_x = tsne.fit_transform(test_x[:n])
    for i in range(n):
        plt.scatter(trans_test_x[i, 0], trans_test_x[i, 1], c='r')
    plt.savefig(output_path + 'before.jpg')
    plt.cla()

    bias = (np.mean(test_x, 0) - np.mean(train_x, 0))
    print('Size of a training feature:', np.linalg.norm(np.mean(train_x, 0), 2))
    print('Size of a test feature:', np.linalg.norm(np.mean(test_x, 0), 2))
    print('Difference of mean: ', np.linalg.norm(bias))
    print('Difference of variance: ', np.linalg.norm(np.var(train_x, 0)-np.var(test_x, 0), 2))

    normalized_test_x = (test_x - np.mean(test_x, 0)) - (np.mean(test_x, 0) - np.mean(train_x, 0))
    normalized_test_x /= np.linalg.norm(normalized_test_x, 2)
    normalized_train_x = train_x - np.mean(train_x, 0)
    normalized_train_x /= np.linalg.norm(normalized_train_x, 2)

    tsne = TSNE(n_components=2, random_state=0)
    trans_normalized_train_x = tsne.fit_transform(normalized_train_x[:n])
    for i in range(n):
        plt.scatter(trans_normalized_train_x[i, 0], trans_normalized_train_x[i, 1], c='g')
    trans_normalized_test_x = tsne.fit_transform(normalized_test_x[:n])
    for i in range(n):
        plt.scatter(trans_normalized_test_x[i, 0], trans_normalized_test_x[i, 1], c='r')
    plt.savefig(output_path + 'after.jpg')
    plt.cla()

    bias = (np.mean(normalized_test_x, 0) - np.mean(normalized_train_x, 0))
    print('Size of a training feature:', np.linalg.norm(np.mean(normalized_train_x, 0), 2))
    print('Size of a test feature:', np.linalg.norm(np.mean(normalized_test_x, 0), 2))
    print('Difference of mean: ', np.linalg.norm(bias))
    print('Difference of variance: ', np.linalg.norm(np.var(normalized_train_x, 0) - np.var(normalized_test_x, 0), 2))

