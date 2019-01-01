import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from chainer.datasets import TupleDataset
from gensim.models import word2vec

def load_imdb_data() -> object:
    pos_pathes = glob("data/pos/*.txt")
    neg_pathes = glob("data/neg/*.txt")
    pathes = pos_pathes + neg_pathes
    labels = [[1] * len(pos_pathes), [0] * len(neg_pathes)]
    return Data("imdb", pathes, labels).load()


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
        assert len(txt_path_list) != 0, "no path in list"
        self.dataname = dataname
        self.txt_path_list = txt_path_list
        assert len(labels) != 0, "no data in labels"
        self.label_list = labels
        self.padding_word = padding_word
        return

    def __call__(self) -> None:
        self.get_info()
        return

    def load(self) -> object:
        """
        Main data-loading Function
        """
        self.load_text()
        self.padding_words()
        self.embed()
        self.build_data()
        return self

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

    def load_text(self, path_list=None) -> None:
        """
        load txt files and build vocab dictionaries and datasets.

        vocabulary (word2index, index2word) : dict
        - mapping from word to index (order of appearance in sentences).
        - ex. {'<PAD/>': 0, 'Hello': 1, 'World': 2}
        """
        if path_list is None:
            path_list = self.txt_path_list

        index2word = {}
        word2index = {}
        counts = collections.Counter()
        sentences = []
        for file in tqdm(path_list, desc="Load Files.."):
            sentence = []
            with open(file) as f:
                for line in f:
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
        return

    def padding_words(self, padding_word="<PAD/>") -> None:
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
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        self.sentences = np.array(padded_sentences)
        return

    def clean_str(string: str) -> str:
        """
        Remove some simbols.
        """
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

    def embed(self, num_workers=2, vectorize_dim=50,
              downsampling=1e-3, context=10, min_word_count=1) -> None:
        """
        Embedding by gensim.Word2Vec skip-gram model and extract weight vector.
        
        Parameters
        ----------
        
        """
        print("Now Training Word2vec model")
        model = word2vec.Word2Vec(self.sentences, workers=num_workers,
                                  size=vectorize_dim,
                                  min_count=min_word_count,
                                  window=context, sample=downsampling)

        self.embedding_weights = {key: model[word] if word in model
                                  else np.random.uniform(-0.25, 0.25,
                                                         model.vector_size)
                                  for key, word in self.word2index.items()}
        print("got embedding weights!")
        return

    def build_data(self, ratio=0.5, seed=0):
        d = [[self.word2index[w] for w in s] for s in self.sentences]
        np.random.seed(seed)
        np.random.shuffle(d)
        np.random.seed(seed)
        np.random.shuffle(self.labels)
        self.x_train, self.x_test = np.split(d, [int(d.size * ratio)])
        self.y_train, self.y_test = np.split(self.labels,
                                             [int(self.labels.size * ratio)])
        return

                                                                                