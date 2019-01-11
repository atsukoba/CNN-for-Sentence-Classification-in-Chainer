import argparse
import numpy as np
import pandas as pd
import chainer
from tqdm import tqdm
from glob import glob
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O
from chainer.training import extensions
# my modules
import models
import data_builder

"""cnn_sentence

sentence classification by CNN on Chainer v.5
"""


def sample_train(data, model_type="CNN-rand") -> object:

    print(data.get_info())
    # build cnn model
    model = L.Classifier(models.cnn[model_type](
        embed_weights=data.embed_weights, n_vocab=data.n_vocab))

    train, test = data.get_chainer_dataset()
    train_iter = iterators.SerialIterator(train, 50)
    test_iter = iterators.SerialIterator(test, 50, repeat=False, shuffle=False)
    optimizer = O.Adam().setup(model)
    updater = training.StandardUpdater(train_iter, optimizer, device=-1)

    # build trainer
    trainer = training.Trainer(updater, (5, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
    trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'elapsed_time', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    chainer.config.train = True
    trainer.run()
    return model.predictor


def skipgram_embedding(data, dim = 50, batchsize = 32, window = 10,
                       negative_sample = 5, epochs = 10) -> list:

    cs = [data.counts[w] for w in range(len(data.counts))]
    loss_func = L.NegativeSampling(dim, cs, negative_sample)
    model = models.SkipGram(data.n_vocab, dim, loss_func)

    # Set up an optimizer
    optimizer = O.Adam()
    optimizer.setup(model)

    # Set up an iterator
    train_iter = models.WindowIterator(data.x_train, window, batchsize)
    val_iter = models.WindowIterator(data.x_test, window, batchsize, repeat=False)

    # Set up an updater
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=-1)

    # Set up a trainer
    trainer = training.Trainer(updater, (epochs, 'epoch'), out="result")
    trainer.extend(extensions.Evaluator(val_iter, model, converter=convert, device=-1))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())
    chainer.config.train = True
    trainer.run()
    return model.predictor.embed.W.data


def convert(batch, device):
    center, contexts = batch
    if device >= 0:
        center = cuda.to_gpu(center)
        contexts = cuda.to_gpu(contexts)
    return center, contexts


if __name__ == "__main__":
    arg = argparse.ArgumentParser(
        description="Train CNN-Sentence-Classification-in-Chainer model.")

    # data parameter
    arg.add_argument("-n", "--name", type=str,
        help="Trainer model and Dataset name")
    arg.add_argument("-t", "--type", type=str,
        help="CNN model type. Choose one from \
        `CNN_rand`, `CNN_static`, `CNN_non_static`, `CNN_multi_ch`",
        default="CNN_non_static")
    arg.add_argument("-d", '--data', type=str,
        help="Path to .txt data", default="data/")
    arg.add_argument('--csv', type=str,
        help="Path to .csv label data. comma separated one-line data",
        default="data/labels.csv")
    
    # Embedding parameter
    arg.add_argument('--w2v_num_workers', type=int,
        help="Word2Vec Param", default=2)
    arg.add_argument('--w2v_vectorize_dim', type=int,
        help="Word2Vec Param", default=300)
    arg.add_argument('--w2v_context', type=int,
        help="Word2Vec Param", default=0)
    # CNN model parameter
    arg.add_argument('--n_filters', type=int,
        help="Height (window) of convolution filter.", default=100)
    arg.add_argument('--hidden_dim', type=int,
        help="Dimension of hidden full-connected layer.", default=50)
    arg.add_argument('--dropout', type=float,
        help="Dropout Ratio", default=.3)
    # training parameter
    arg.add_argument('--batch', type=int,
        help="Size of batch", default=50)
    arg.add_argument('--max_epoch', type=int,
        help="Num of epochs", default=10)
    arg.add_argument('--gpu_id', type=int,
        help="gpu id, when using CPU, should be set -1", default=-1)
    arg.add_argument('--opt',
        help="Optimization function of NN", default="Adam")
    arg.add_argument('--test', type=bool,
        help="Do test or not after training", default=True)
    arg = arg.parse_args()

    # read data
    txtpathes = glob(arg.data + "*txt")
    txtpathes.extend(glob(arg.data + "*pos"))
    txtpathes.extend(glob(arg.data + "*neg"))
    labels = np.loadtxt(arg.csv, delimiter=",", dtype="str")
    print("len of .txt file pathes: ", len(txtpathes))
    print("len of .csv label data", len(labels))
    if len(txtpathes) == len(set(labels)):
        read_line=True
    else:
        # data check
        assert len(txtpathes) == len(labels),\
            "Length of file pathes and labels should be same"
        read_line=False
    # make data object
    data = data_builder.Data(arg.name, txtpathes, labels,
        w2v_num_workers=arg.w2v_num_workers,
        w2v_vectorize_dim=arg.w2v_vectorize_dim,
        w2v_context=arg.w2v_context).load(docs_line_style=read_line)
    # print data infomation
    print(data.get_info())
    # embedding
    data.embed()
    # build cnn model
    model = L.Classifier(models.cnn[model_type](
        embed_weights=data.embed_weights, n_vocab=data.n_vocab,
        n_filters=arg.n_filters, hidden_dim=arg.hidden_dim, dropout=arg.dropout))

    train, test = data.get_chainer_dataset()
    train_iter = iterators.SerialIterator(train, arg.batch)
    test_iter = iterators.SerialIterator(test, arg.batch,
                                         repeat=False, shuffle=False)

    eval("optimizer = O.{}().setup(model)".format(arg.opt))

    updater = training.StandardUpdater(train_iter, optimizer,
                                       device=arg.gpu_id)

    # build trainer
    trainer = training.Trainer(updater, (5, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
    trainer.extend(extensions.snapshot(),
                   trigger=(arg.max_epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'elapsed_time', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    chainer.config.train = True
    trainer.run()
