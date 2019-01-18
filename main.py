import argparse
from File import TrainFile, TestFile
import numpy as np
import torch
import models
import sklearn


def parse_arg():

    parser = argparse.ArgumentParser()

    # training
    parser.add_argument('-model', type=str, default='mlp')
    parser.add_argument('-bs', type=int, default=int(1e3))
    parser.add_argument('-lr', type=float, default=5e-3)
    parser.add_argument('-lr_decay', action='store_true', default=False)
    parser.add_argument('-optim', type=str, default='adam')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-max_epoch_num', type=int, default=10)
    parser.add_argument('-stop', type=int, default=-1)
    parser.add_argument('-f', type=int, default=5)
    parser.add_argument('-num_seed', type=int, default=10)

    # svm params
    parser.add_argument('-kernel', type=str, default='linear')
    parser.add_argument('-c', type=float, default=1)

    # LR params
    parser.add_argument('-lamb', type=float, default=0)

    # KNN params
    parser.add_argument('-k', type=int, default=20)

    # MLP params
    parser.add_argument('-ac_fn', type=str, default='relu')
    parser.add_argument('-no_valid', action='store_true', default=False)
    parser.add_argument('-dr', type=float, default=0.5)
    # parser.add_argument('-lamb', type=float, default=0) # the same as that in lr

    # XgBoost params
    parser.add_argument('-max_depth', type=int, default=6)
    parser.add_argument('-eta', type=float, default=0.3)
    parser.add_argument('-sub_sample', type=float, default=0.5)
    parser.add_argument('-nthread', type=int, default=4)
    # parser.add_argument('-lamb', type=float, default=0) # the same as that in lr

    # NaiveBayes params
    parser.add_argument('-model_type', type=str, default='multinomial')

    # IO
    parser.add_argument('-gpu', type=int, default=-1)
    parser.add_argument('-input_path', type=str, default='/home/lhchen/nas/statlearning/data/')
    parser.add_argument('-train_file', type=str, default='train.csv')
    parser.add_argument('-test_file', type=str, default='test.csv')
    parser.add_argument('-output_path', type=str, default='/home/lhchen/nas/res/stat/')
    parser.add_argument('-output_file', type=str, default='res.csv')
    parser.add_argument('-note', type=str, default='test')
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
                           args.max_epoch_num, args.stop, args.bs, args.lamb, args.f, test_data.write, args.seed,
                           args.num_seed, args.lr_decay,)
        data = model.train_model(train_data, args.no_valid)
    elif args.model == 'xgb':
        model = models.XgBoost(args.max_depth, args.eta, args.nthread, args.lamb, args.sub_sample)
        model.train_model(train_data, args.max_epoch_num, args.no_valid)
    else:
        raise NotImplementedError

    print('predicting')
    preds = model.predict(test_data.feat)
    test_data.output(['id', 'categories'], preds)
