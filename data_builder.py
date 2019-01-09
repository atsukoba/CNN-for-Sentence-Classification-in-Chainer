import re
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from chainer.datasets import TupleDataset
from gensim.models import Word2Vec


class Data:
    """
    Class of text data, embedding weights.

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
                 txt_path_list: list, labels: list,
                 padding_word="<PAD/>") -> None:
        """
        set some parameters.
        """
        assert len(txt_path_list) != 0, "No path in list"
        self.dataname = dataname
        self.txt_path_list = txt_path_list
        assert len(labels) != 0, "No data in labels"
        self.labels = np.array(labels)
        self.padding_word = padding_word
        return

    def __call__(self) -> None:
        self.get_info()
        return

    def load(self, docs_line_style=False) -> object:
        """
        Main data-loading Function
        """
        if docs_line_style:
            self.load_docs_line()
        else:
            self.load_docs_file()
        self.pad_sentences()
        self.build_data()
        return self

    def get_info(self) -> str:
        output = "Data Info {}\n".format(self.dataname)
        output += "-" * 30 + "\n"
        output += "Vocab: {}\n".format(self.n_vocab)
        output += "Sentences: {}\n".format(len(self.sentences))
        output += "-" * 30 + "\n"
        output += "x_train: {}\n".format(self.x_train.shape)
        output += "x_test: {}\n".format(self.x_test.shape)
        output += "y_train: {}\n".format(self.y_train.shape)
        output += "y_test: {}\n".format(self.y_test.shape)
        return output

    def get_chainer_dataset(self) -> "chainer.datasets.Tupledataset":
        """
        get train/test splited chainer.datasets.Tupledataset data
        call this after exec self.build_data()
        """
        return (TupleDataset(self.x_train, self.y_train),
                TupleDataset(self.x_test, self.y_test))

    def load_docs_file(self, path_list=None) -> None:
        """
        load txt files and build vocab dictionaries and datasets.

        data format example
        -------------------
        - doc1.txt
            doc1 strings...
        - doc2.txt
            doc2 strings...

        variables
        ---------
        vocabulary (word2index, index2word) : dict
        - mapping from word to index (order of appearance in sentences).
        - ex. {'<PAD/>': 0, 'Hello': 1, 'World': 2}
        """
        if path_list is None:
            path_list = self.txt_path_list

        index2word = {}
        word2index = {}
        word2index[self.padding_word] = 0
        index2word[0] = self.padding_word
        counts = collections.Counter()
        sentences = []
        for file in tqdm(path_list, desc="Read Files.."):
            sentence = []
            with open(file, encoding="utf-8") as f:
                for line in f:
                    line = self.clean_str(line)
                    for word in line.split():
                        if word not in word2index:
                            ind = len(word2index)
                            word2index[word] = ind
                            index2word[ind] = word
                        counts[word2index[word]] += 1
                        sentence.append(word)
            sentences.append(sentence)
        self.index2word = index2word
        self.word2index = word2index
        self.sentences = sentences
        self.counts = counts
        self.n_vocab = len(word2index)
        return

    def load_docs_line(self, path_list=None) -> None:
        """
        load txt files and build vocab dictionaries and datasets.
        set different labels along with files.
        data format
        -----------
        - data.txt
            doc1 \n
            doc2 \n
            : \n
            : \n
            docN

        variables
        ---------
        vocabulary (word2index, index2word) : dict
        - mapping from word to index
        - ex. {'<PAD/>': 0, 'Hello': 1, 'World': 2}
        """
        if path_list is None:
            path_list = self.txt_path_list

        index2word = dict()
        word2index = dict()
        word2index[self.padding_word] = 0
        index2word[0] = self.padding_word
        counts = collections.Counter()
        sentences = list()
        labels = list()
        for label, file in tqdm(enumerate(path_list),
                                desc="Read lines.."):
            with open(file, encoding="utf-8") as f:
                for line in f:
                    sentence = []
                    line = self.clean_str(line)
                    for word in line.split():
                        if word not in word2index:
                            ind = len(word2index)
                            word2index[word] = ind
                            index2word[ind] = word
                        counts[word2index[word]] += 1
                        sentence.append(word)
                    sentences.append(sentence)
                    labels.append(label)
        self.index2word = index2word
        self.word2index = word2index
        self.sentences = np.array(sentences)
        self.labels = np.array(labels)
        self.counts = counts
        self.n_vocab = len(word2index)
        return
    
    def pad_sentences(self) -> None:
        """
        Pads all sentences to the same length. The length is defined by
        the longest sentence. Returns padded sentences, in order to align
        matrix dimensions.

        Parameters
        ----------
        sentences : list of lists of words
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
        sequence_length = max(len(s) for s in self.sentences)
        padded_sentences = []
        for sentence in tqdm(self.sentences, desc="Padding"):
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [self.padding_word] * num_padding
            padded_sentences.append(new_sentence)
        self.raw_sentences = self.sentences
        self.sentences = np.array(padded_sentences)
        return

    def clean_str(self, string: str) -> str:
        """
        Remove some simbols and extend abbreviated expressions.
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " is ", string)
        string = re.sub(r"\'ve", " have ", string)
        string = re.sub(r"n\'t", " not ", string)
        string = re.sub(r"\'re", " are ", string)
        string = re.sub(r"\'d", " would ", string)
        string = re.sub(r"\'ll", " will ", string)
        string = re.sub(r",", " ", string)
        string = re.sub(r"!", " ", string)
        string = re.sub(r"\(", " ", string)
        string = re.sub(r"\)", " ", string)
        string = re.sub(r"\?", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def build_data(self, ratio=0.5, seed=0) -> None:
        """
        Build train-test-splited X and y(label) data.
        Call this after executing self.load()
        
        data x dim : (n_sample, n_channel, words)
        """
        # shuffle data
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.labels = self.labels[shuffle_indices]
        # make index dataset 
        d = np.array([[int(self.word2index[w]) for w in s]
                      for s in self.sentences])[shuffle_indices]
        # split data
        split = int(len(d) * ratio)
        self.x_train, self.x_test = (np.array(d[:split])[:, np.newaxis, :],
                                     np.array(d[split:])[:, np.newaxis, :])
        self.y_train, self.y_test = (np.array(self.labels[:split]),
                                     np.array(self.labels[split:]))

        return
    
    def embed(self, num_workers=2, vectorize_dim=50,
              downsampling=1e-3, context=10, min_word_count=1) -> None:
        """
        Embedding by gensim.Word2Vec skip-gram model and extract weight vector.

        Parameters
        ----------
        num_workers: int
        vectorize_dim: int
        downsampling: float
        context: int
        min_word_count: int
        """
        print("Training Word2vec model...")
        self.embed_model = Word2Vec(self.raw_sentences,
                                    workers=num_workers,
                                    size=vectorize_dim,
                                    min_count=min_word_count,
                                    window=context, sample=downsampling)
        self.embed_model.init_sims(replace=True)
        self.embed_weights = np.array([self.embed_model[word]\
            if word in self.embed_model else np.random.uniform(-0.25, 0.25,
            self.embed_model.vector_size)
            for key, word in self.word2index.items()])
                                                                 
        print("got embedding weights!")
        return


def load_imdb_data(pos_dir="data/pos/", neg_dir="data/neg/",
                   file_extend="txt") -> object:
    """
    load sample text data and build vocalbs. datasets, embedding-weights.
    """
    pos_pathes = glob(pos_dir + "*" + file_extend)
    neg_pathes = glob(pos_dir + "*" + file_extend)
    pathes = pos_pathes + neg_pathes
    labels = [1] * len(pos_pathes) + [0] * len(neg_pathes)
    return Data("imdb", pathes, labels).load()


def load_pos_neg_file(data_dir="data/", file_extends=["pos", "neg"]) -> object:
    """
    load sample text data and build vocalbs. datasets, embedding-weights.
    .pos/.neg style text file as data.
    """
    pathes = []
    for ext in file_extends:
        l = glob(data_dir + "/*" + ext)
        pathes.extend(l)
    labels = ["dummy", "dummy"]
    return Data("imdb", pathes, labels).load(docs_line_style=True)


if __name__ == "__main__":
    pass
