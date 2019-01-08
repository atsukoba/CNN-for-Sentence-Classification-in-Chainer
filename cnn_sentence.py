import argparse
import numpy as np
import chainer
# my modules
import models
import data_builder
from chainer.backends import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
from chainer import training
from chainer.training import extensions


def sample_train(model_type="non-static"):

    # load imdb data
    data = data_builder.load_imdb_data()
    print(data.get_info())

    # embedding
    data.embed()

    # build cnn model
    model = L.Classifier(models.CNN_non_static(
        embedding_weight=data.embedding_weights))

    train, test = data.get_chainer_dataset()
    train_iter = chainer.iterators.SerialIterator(train, 64)
    test_iter = chainer.iterators.SerialIterator(test, 64, repeat=False, shuffle=False)
    optimizer = chainer.optimizers.Adam().setup(model)
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
    
    return


if __name__ == "__main__":
    pass
