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


def train(model_type="non-static"):

    data = data_builder.load_imdb_data()
    model = L.Classifier(models.CNN_non_static(
        embedding_weight=data.embedding_weights))

    return


if __name__ == "__main__":
    pass
