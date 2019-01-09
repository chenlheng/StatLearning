import argparse
from File import TrainFile, TestFile
import numpy as np
import torch
import models


def parse_arg():

    parser = argparse.ArgumentParser()

    # training
    parser.add_argument('-model', type=str, default='linearsvm')
    parser.add_argument('-bs', type=int, default=int(5e2))
    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('-lr_decay', action='store_true', default=False)
    parser.add_argument('-optim', type=str, default='sgd')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-max_epoch_num', type=int, default=10)

    # svm params
    parser.add_argument('-lamb', type=float, default=0.1)

    # EM params
    parser.add_argument('-max_step', type=int, default=int(1e3))

    # KNN params
    parser.add_argument('-k', type=int, default=20)
    parser.add_argument('-f', type=int, default=10)

    # IO
    parser.add_argument('-gpu', type=str, default='-1')
    parser.add_argument('-input_path', type=str, default='/home/lhchen/nas/statlearning/data/')
    parser.add_argument('-train_file', type=str, default='train.csv')
    parser.add_argument('-test_file', type=str, default='test.csv')
    parser.add_argument('-output_path', type=str, default='/home/lhchen/nas/res/stat/')
    parser.add_argument('-output_file', type=str, default='_118033910029.csv')
    parser.add_argument('-note', type=str, default='test')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_arg()

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_file = args.input_path + args.train_file
    train_data = TrainFile(train_file)

    if args.optim == 'sgd':
        optim = torch.optim.SGD
    else:
        raise NotImplementedError

    print('training')
    if args.model == 'linearsvm':
        model = models.LinearSVM(train_data.feat_dim, optim, args.lr)
        model.train_model(train_data, args.max_epoch_num, args.lr, args.bs, args.lamb, args.lr_decay)
    if args.model == 'knn':
        model = models.KNN(args.k, args.f)
        model.train_model(train_data)
    else:
        raise NotImplementedError

    test_file = args.input_path + args.test_file
    output_path = args.output_path + args.note + '/'
    output_file = args.output_path + args.note + '/' + args.output_file
    test_data = TestFile(test_file, output_path, output_file)

    print('predicting')
    preds = model.predict(test_data)
    test_data.output(['id', 'categories'], preds)
