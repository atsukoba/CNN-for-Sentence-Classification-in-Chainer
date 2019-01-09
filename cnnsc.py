import argparse
import numpy as np
import chainer
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
        embed_weights=data.embed_weights,
        conv_filter_windows=[3, 8], n_vocab=data.n_vocab))

    train, test = data.get_chainer_dataset()
    train_iter = iterators.SerialIterator(train, 64)
    test_iter = iterators.SerialIterator(test, 64, repeat=False, shuffle=False)
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
    pass
