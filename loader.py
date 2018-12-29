import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from chainer.datasets import TupleDatasets


class Data:
    """
    Parameters
    ----------
    dataname : str
        dataname, set as you like.
    txt_path_list : list
        list of pathes.
        ["foo.txt", "bar.txt", ..., "woo.txt"]
    labels : list
        list of labels.
    """
    def __init__(self, dataname: str,
                 txt_path_list: list, labels: list) -> None:
        assert len(txt_path_list) != 0, "no path in list"
        self.dataname = dataname
        self.txt_path_list = txt_path_list

        assert len(labels) != 0, "no data in labels"
        self.label_list = labels

        return

    def __call__(self) -> None:
        self.info()
        return

    def load(self) -> None:
        """
        Main data-load Function
        """
        return

    def get_info(self) -> str:
        output = "Data Info {}\n".format(self.dataname)
        output += "-" * 30 + "\n"
        output += "x_train: {}\n".format(self.x_train.shape)
        output += "x_test: {}\n".format(self.x_test.shape)
        output += "y_train: {}\n".format(self.y_train.shape)
        output += "y_test: {}\n".format(self.y_test.shape)
        return output

    def get_chainer_dataset(self):
        return

    def _load_text(self) -> None:
        """
        load txt files and make dictionaries and datasets.
        """
        index2word = {}
        word2index = {}
        counts = collections.Counter()
        dataset = []
        for file in tqdm(self.txt_path_list):
            with open(file) as f:
                for line in f:
                    for word in line.split():
                        if word not in word2index:
                            ind = len(word2index)
                            word2index[word] = ind
                            index2word[ind] = word
                        counts[word2index[word]] += 1
                        dataset.append(word2index[word])
        self.index2word = index2word
        self.word2index = word2index
        self.dataset = dataset
        return

    def _padding_words(self):
        """
        """
        return
