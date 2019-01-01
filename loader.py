import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from chainer.datasets import TupleDataset


def load_imdb_data() -> object:
    pos_pathes = glob("data/pos/*.txt")
    neg_pathes = glob("data/neg/*.txt")
    pathes = pos_pathes + neg_pathes
    labels = [[1] * len(pos_pathes), [0] * len(neg_pathes)]
    return Data("imdb", pathes, labels)

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
        self._load_text()
        
        return

    def get_info(self) -> str:
        output = "Data Info {}\n".format(self.dataname)
        output += "-" * 30 + "\n"
        output += "Vocab: {}\n".format(len(self.index2word))
        output += "Sentences: {}\n".format(len(self.sentences))
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
        load txt files and build vocab dictionaries and datasets.

        vocabulary (word2index, index2word) : dict
            mapping from word to index (order of appearance in sentences).
            ex. {'<PAD/>': 0, 'Hello': 1, 'World': 2}
        """
        index2word = {}
        word2index = {}
        counts = collections.Counter()
        sentences = []
        for file in tqdm(self.txt_path_list):
            sentence = []
            with open(file) as f:
                for line in f:
                    for word in line.split():
                        if word not in word2index:
                            ind = len(word2index)
                            word2index[word] = ind
                            index2word[ind] = word
                        counts[word2index[word]] += 1
                    sentence.append(line)
            sentences.append(sentence)
        self.index2word = index2word
        self.word2index = word2index
        self.sentences = sentences
        self.counts = counts
        return

    def _padding_words(self):
        """
        Pads all sentences to the same length. The length is defined by
        the longest sentence. Returns padded sentences, in order to align
        matrix dimensions.

        Parameters
        ----------
        sentences : list of list
            [[sentence1], [sentence2], ..., [sentenceN]]
            Each sentence is list of words.

        padding_word : str
            Fill missed length of string with padding_word.
            When sequence length of longest sentence is 4,
            sentence=["Hello", "World"]
            would be padded_sentence=["Hello", "World", "<PAD/>", "<PAD/>"].

        padded_sentences : list of str
            [[padded_sentence1], [padded_sentence2], ..., [padded_sentenceN]]
            All the sentences have same char string length.
        """
        
        return

    def _clean_str(string: str) -> str:
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
