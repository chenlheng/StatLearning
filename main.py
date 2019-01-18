import argparse
from File import TrainFile, TestFile
import numpy as np
import torch
import models
import sklearn


def parse_arg():

    parser = argparse.ArgumentParser()

    # training
    parser.add_argument('-model', type=str, default='mlp',
                        help='name of used model')
    parser.add_argument('-seed', type=int, default=0,
                        help='random seed for training on training dataset')
    parser.add_argument('-f', type=int, default=5,
                        help='number of folds in CV')
    parser.add_argument('-num_seed', type=int, default=10,
                        help='number of repeated experiments with different seeds')
    parser.add_argument('-no_valid', action='store_true', default=False,
                        help='call to disable CV')

    # svm params
    parser.add_argument('-kernel', type=str, default='linear',
                        help='name of used kernel')
    parser.add_argument('-c', type=float, default=1,
                        help='penalty coef on error term')

    # LR params
    parser.add_argument('-lamb', type=float, default=0,
                        help='coef of l2-reguralization')

    # KNN params
    parser.add_argument('-k', type=int, default=20,
                        help='number of nearest neighbors')

    # MLP params
    parser.add_argument('-ac_fn', type=str, default='relu',
                        help='activation function')
    parser.add_argument('-dr', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('-max_epoch_num', type=int, default=100,
                        help='max number of training epoches')
    parser.add_argument('-bs', type=int, default=int(1e3),
                        help='batch size')
    parser.add_argument('-lr', type=float, default=5e-3,
                        help='learning rate')
    parser.add_argument('-lr_decay', action='store_true', default=False,
                        help='call to enable learning rate decaying')
    parser.add_argument('-optim', type=str, default='adam',
                        help='name of optimizer used')
    # parser.add_argument('-lamb', type=float, default=0) # the same as that in lr

    # NaiveBayes params
    parser.add_argument('-model_type', type=str, default='multinomial',
                        help='name of kernel used')

    # IO
    parser.add_argument('-gpu', type=int, default=-1)
    parser.add_argument('-input_path', type=str, default='/home/lhchen/nas/statlearning/data/')
    parser.add_argument('-train_file', type=str, default='train.csv')
    parser.add_argument('-test_file', type=str, default='test.csv')
    parser.add_argument('-output_path', type=str, default='/home/lhchen/nas/res/stat/')
    parser.add_argument('-output_file', type=str, default='res.csv')
    parser.add_argument('-note', type=str, default='test',
                        help='name of folder to store predictions')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_arg()

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    np.random.seed(0)
    torch.manual_seed(0)

    train_file = args.input_path + args.train_file
    train_data = TrainFile(train_file)

    test_file = args.input_path + args.test_file
    output_path = args.output_path + args.note + '/'
    output_file = args.output_path + args.note + '/' + args.output_file
    test_data = TestFile(test_file, train_data, output_path, output_file)

    if args.optim == 'sgd':
        optim = torch.optim.SGD
    elif args.optim == 'adam':
        optim = torch.optim.Adam
    elif args.optim == 'adagrad':
        optim = torch.optim.Adagrad
    else:
        raise NotImplementedError

    print('training')
    if args.model == 'knn':
        model = models.KNN(args.k, args.f, test_data.write, args.seed, args.num_seed)
        model.train_model(train_data, args.no_valid)
    elif args.model == 'lr':
        model = models.RidgeRegression(args.lamb, args.f, test_data.write, args.seed, args.num_seed)
        model.train_model(train_data, args.no_valid)
    elif args.model == 'nb':
        model = models.NaiveBayes(args.model_type, args.f, test_data.write, args.seed, args.num_seed)
        model.train_model(train_data, args.no_valid)
    elif args.model == 'svm':
        model = models.SVM(args.kernel, args.c, args.f, test_data.write, args.seed, args.num_seed)
        model.train_model(train_data, args.no_valid)
    elif args.model == 'mlp':
        model = models.MLP(train_data.feat_dim, train_data.cat_num, optim, args.ac_fn, args.dr, args.lr, args.gpu,
                           args.max_epoch_num, args.bs, args.lamb, args.f, test_data.write, args.seed,
                           args.num_seed, args.lr_decay,)
        data = model.train_model(train_data, args.no_valid)
    else:
        raise NotImplementedError

    print('predicting')
    preds = model.predict(test_data.feat)
    test_data.output(['id', 'categories'], preds)
