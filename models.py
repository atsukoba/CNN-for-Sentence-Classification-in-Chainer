import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Chain
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

"""
>> (on the paper)
>> 3.1 Hyperparameters and Training
>> For all datasets we use: rectified linear units, filter
>> windows (h) of 3, 4, 5 with 100 feature maps each,
>> dropout rate (p) of 0.5, l2 constraint (s) of 3, and
>> mini-batch size of 50. These values were chosen
>> via a grid search on the SST-2 dev set.
"""


class CNN_rand(Chain):
    """
    Chain of CNN for Sentence classification model.
    """
    def __init__(self, conv_filter_windows=[3, 4, 5],
                 n_vocab: int, embed_weights=None,
                 embed_dim=50, n_filters=100,
                 hidden_dim=50, n_labels=2):
        self.embed_dim = embed_dim
        w = np.random.rand(n_vocab, embed_dim)
        super(CNN_rand, self).__init__()
        with self.init_scope():
            # Embedding
            self.embed = L.EmbedID(n_vocab, embed_dim,
                                   initialW=w)
            # Convolutions
            self.convs = list()
            for window in conv_filter_windows:
                self.convs.append(
                    L.Convolution2D(1, n_filters, ksize=(window, embed_dim)))
            # full connection
            self.fc4 = L.Linear(None, hidden_dim)
            self.fc5 = L.Linear(hidden_dim, n_labels)

    def __call__(self, x):
        x = self.embed(x)
        conved = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.average_pooling_2d(h, (2, self.embed_dim))
            conved.append(h)
        # concatenate along conved dimention (axis=2)
        x = F.concat(conved, axis=2)
        x = F.dropout(F.relu(self.fc4(x)), 0.5)
        if chainer.config.train:
            return self.fc5(x)
        return F.softmax(self.fc5(x))


class CNN_static(Chain):
    """
    Chain of CNN for Sentence classification model.
    """
    def __init__(self, conv_filter_windows=[3, 4, 5],
                 embed_weights: list, n_vocab: int,
                 embed_dim=50, hidden_dim=50,
                 n_labels=2):
        self.embed_weights = embed_weights
        self.embed_dim = embed_dim
        super(CNN_static, self).__init__()
        with self.init_scope():
            # Convolutions
            self.convs = list()
            for window in conv_filter_windows:
                self.convs.append(
                    L.Convolution2D(1, n_filters, ksize=(window, embed_dim)))
            # full connection
            self.fc4 = L.Linear(None, hidden_dim)
            self.fc5 = L.Linear(hidden_dim, n_labels)

    def __call__(self, x):
        x = F.embed_id(x, initialW=self.embed_weights)
        conved = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.average_pooling_2d(h, (2, self.embed_dim))
            conved.append(h)
        # concatenate along conved dimention (axis=2)
        x = F.concat(conved, axis=2)
        x = F.dropout(F.relu(self.fc4(x)), 0.5)
        if chainer.config.train:
            return self.fc5(x)
        return F.softmax(self.fc5(x))


class CNN_non_static(Chain):
    """
    Chain of CNN for Sentence classification model.
    """
    def __init__(self, conv_filter_windows=[3, 4, 5],
                 embed_weights: list, n_vocab: int,
                 embed_dim=50, hidden_dim=50,
                 n_labels=2):
        self.embed_dim = embed_dim
        super(CNN_non_static, self).__init__()
        with self.init_scope():
            # Embedding
            self.embed = L.EmbedID(n_vocab, embed_dim,
                                   initialW=embed_weights,
                                   ignore_label=-1)
            # Convolutions
            self.convs = list()
            for window in conv_filter_windows:
                self.convs.append(
                    L.Convolution2D(1, n_filters, ksize=(window, embed_dim)))
            # full connection
            self.fc4 = L.Linear(None, hidden_dim)
            self.fc5 = L.Linear(hidden_dim, n_labels)

    def __call__(self, x):
        x = self.embed(x)
        conved = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.average_pooling_2d(h, (2, self.embed_dim))
            conved.append(h)
        # concatenate along conved dimention (axis=2)
        x = F.concat(conved, axis=2)
        x = F.dropout(F.relu(self.fc4(x)), 0.5)
        if chainer.config.train:
            return self.fc5(x)
        return F.softmax(self.fc5(x))


class CNN_multi_ch(Chain):
    """
    Chain of CNN for Sentence classification model.
    """
    def __init__(self, conv_filter_windows=[3, 4, 5],
                 embed_weights: list, n_vocab: int,
                 embed_dim=50, hidden_dim=50,
                 n_labels=2):
        self.embed_dim = embed_dim
        super(CNN_multi_ch, self).__init__()
        with self.init_scope():
            # Embedding
            self.embed = L.EmbedID(n_vocab, embed_dim,
                                   initialW=embed_weights)
            # Convolutions
            self.convs = list()
            for window in conv_filter_windows:
                self.convs.append(
                    L.Convolution2D(1, n_filters, ksize=(window, embed_dim)))
            # full connection
            self.fc4 = L.Linear(None, hidden_dim)
            self.fc5 = L.Linear(hidden_dim, n_labels)

    def __call__(self, x):
        x1 = F.embed_id(x, initialW=self.embed_weights)
        x2 = self.embed(x)
        # concatenate along channel dimention (axis=1)
        x = F.concat([x1, x2], axis=1)
        conved = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.average_pooling_2d(h, (2, self.embed_dim))
            conved.append(h)
        # concatenate along conved dimention (axis=2)
        x = F.concat(conved, axis=2)
        x = F.dropout(F.relu(self.fc4(x)), 0.5)
        if chainer.config.train:
            return self.fc5(x)
        return F.softmax(self.fc5(x))


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


# call these classes like `model = models.cnn["MODEL_NAME"](PARAMS)`
cnn = {"CNN_rand": CNN_rand,
       "CNN_static": CNN_static,
       "CNN_non_static": CNN_non_static,
       "CNN_multi_ch": CNN_multi_ch}


if __name__ == "__main__":
    print("import this module or exec from cnnsc.py")
