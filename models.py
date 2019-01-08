import numpy as np
import chainer
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer.training import extensions

"""
(on the paper)
3.1 Hyperparameters and Training
For all datasets we use: rectified linear units, filter
windows (h) of 3, 4, 5 with 100 feature maps each,
dropout rate (p) of 0.5, l2 constraint (s) of 3, and
mini-batch size of 50. These values were chosen
via a grid search on the SST-2 dev set.
"""


class CNN_rand(Chain):
    """
    Chain of CNN for Sentence classification model.
    """
    def __init__(self, conv_filter_windows: list,
                 n_vocab: int,
                 embed_dim=50, hidden_dim=50,
                 n_labels=2):
        w = np.random.rand(n_vocab, embed_dim)
        super(CNN_rand, self).__init__()
        with self.init_scope():
            # Embedding
            self.embed = L.EmbedID(n_vocab, embed_dim,
                                   initialW=w)
            # Convolutions
            self.convs = list()
            for w in conv_filter_windows:
                self.convs.append(
                    L.Convolution1D(1, 1, ksize=w))
            # full connection
            self.fc4 = L.Linear(None, hidden_dim)
            self.fc5 = L.Linear(hidden_dim, 2)

    def __call__(self, x):
        x = self.embed(x)
        conved = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.max_pooling_1d(h, 2)
            h = F.flatten(h)
            conved.append(h)
        x = F.concat(conved)
        x = F.relu(self.fc4(x))
        if chainer.config.train:
            return self.fc5(x)
        return F.softmax(self.fc5(x))


class CNN_static(Chain):
    """
    Chain of CNN for Sentence classification model.
    """
    def __init__(self, weights: list):
        super(CNN_static, self).__init__()
        with self.init_scope():
            self.initial_embed = weights
            # Convolutions
            self.conv2 = L.Convolution2D(
                in_channels=6, out_channels=16, ksize=5, stride=1)
            self.conv3 = L.Convolution2D(
                in_channels=16, out_channels=120, ksize=4, stride=1)
            # full connection
            self.fc4 = L.Linear(None, 84)
            self.fc5 = L.Linear(84, 10)

    def __call__(self, x):
        h = F.embed_id(x)
        h = F.sigmoid(self.conv1(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.sigmoid(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.sigmoid(self.conv3(h))
        h = F.sigmoid(self.fc4(h))
        if chainer.config.train:
            return self.fc5(h)
        return F.softmax(self.fc5(h))


class CNN_non_static(Chain):
    """
    Chain of CNN for Sentence classification model.
    """
    def __init__(self, embedding_weight: list):
        super(CNN_non_static, self).__init__()
        with self.init_scope():
            # Convolutions
            self.embed = L.EmbedID(in_channels=1, out_channels=6, ksize=5, stride=1)
            self.conv2 = L.Convolution2D(
                in_channels=6, out_channels=16, ksize=5, stride=1)
            self.conv3 = L.Convolution2D(
                in_channels=16, out_channels=120, ksize=4, stride=1)
            # full connection
            self.fc4 = L.Linear(None, 84)
            self.fc5 = L.Linear(84, 10)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc4(h))
        if chainer.config.train:
            return self.fc5(h)
        return F.softmax(self.fc5(h))


class CNN_multi_ch(Chain):
    """
    Chain of CNN for Sentence classification model.
    """
    def __init__(self):
        super(CNN_multi_ch, self).__init__()
        with self.init_scope():
            # Convolutions
            self.embed = L.EmbedID()
            self.conv2 = L.Convolution1D()
            self.conv3 = L.Convolution1D()
            # full connection
            self.fc4 = L.Linear(None, 84)
            self.fc5 = L.Linear(84, 10)

    def __call__(self, x):
        h_ch1 = F.sigmoid(F.embed_id(x))
        h_ch2 = F.sigmoid(self.embed(x))
        
        h = F.relu(self.conv1(h_ch1, h_ch2))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc4(h))
        if chainer.config.train:
            return self.fc5(h)
        return F.softmax(self.fc5(h))


class SkipGram(Chain):
    """Definition of Skip-gram Model"""

    def __init__(self, n_vocab, n_units, loss_func):
        super(SkipGram, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(
                n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.loss_func = loss_func

    def forward(self, x, contexts):
        e = self.embed(contexts)
        batch_size, n_context, n_units = e.shape
        x = F.broadcast_to(x[:, None], (batch_size, n_context))
        e = F.reshape(e, (batch_size * n_context, n_units))
        x = F.reshape(x, (batch_size * n_context,))
        loss = self.loss_func(e, x)
        reporter.report({'loss': loss}, self)
        return loss


class WindowIterator(chainer.dataset.Iterator):
    """Dataset iterator to create a batch of sequences at different positions.
    This iterator returns a pair of the current words and the context words.
    """

    def __init__(self, dataset, window, batch_size, repeat=True):
        self.dataset = np.array(dataset, np.int32)
        self.window = window  # size of context window
        self.batch_size = batch_size
        self._repeat = repeat
        # order is the array which is shuffled ``[window, window + 1, ...,
        # len(dataset) - window - 1]``
        self.order = np.random.permutation(
            len(dataset) - window * 2).astype(np.int32)
        self.order += window
        self.current_position = 0
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False

    def __next__(self):
        """This iterator returns a list representing a mini-batch.
        Each item indicates a different position in the original sequence.
        """
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i:i_end]
        w = np.random.randint(self.window - 1) + 1
        offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
        pos = position[:, None] + offset[None, :]
        contexts = self.dataset.take(pos)
        center = self.dataset.take(position)

        if i_end >= len(self.order):
            np.random.shuffle(self.order)
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return center, contexts

    def epoch_detail(self):
        return self.epoch + float(self.current_position) / len(self.order)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)

    
if __name__ == "__main__":
    pass
