import re
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from chainer.datasets import TupleDataset
from gensim.models import Word2Vec


def load_imdb_data() -> object:
    """
    load sample text data and build vocalbs. datasets, embedding-weights.
    """
    pos_pathes = glob("data/pos/*.txt")
    neg_pathes = glob("data/neg/*.txt")
    pathes = pos_pathes + neg_pathes
    labels = [1] * len(pos_pathes) + [0] * len(neg_pathes)
    return Data("imdb", pathes, labels).load()


class Data:
    """
    Class of text data, embedding weights.

    Parameters
    ==========
    dataname : str
        dataname, set as you like.
    txt_path_list : list
        list of pathes.
        ["foo.txt", "bar.txt", ..., "woo.txt"]
    labels : list
        list of labels.
    
    Parameters for Word2Vec embedding
    ---------------------------------
    num_workers: int
    vectorize_dim: int
    downsampling: float
    context: int
    min_word_count: int
    """
    def __init__(self, dataname: str,
                 txt_path_list: list, labels: list,
                 padding_word="<PAD/>",
                 w2v_num_workers=2, w2v_vectorize_dim=300,
                 w2v_downsampling=1e-3, w2v_context=10,
                 w2v_min_word_count=1) -> None:
        """
        set some parameters.
        """
        assert len(txt_path_list) != 0, "no path in list"
        self.dataname = dataname
        self.txt_path_list = txt_path_list
        assert len(labels) != 0, "no data in labels"
        self.labels = np.array(labels)
        self.n_labels = len(set(labels))
        self.padding_word = padding_word
        # embedding parameters
        self.w2v_vectorize_dim = w2v_vectorize_dim
        self.w2v_downsampling = w2v_downsampling
        self.w2v_context = w2v_context
        self.w2v_min_word_count = w2v_min_word_count
        self.w2v_num_workers = w2v_num_workers
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
        word2index[self.padding_word] = 0
        index2word[0] = self.padding_word
        counts = collections.Counter()
        sentences = []
        for file in tqdm(path_list, desc="Read Files.."):
            sentence = []
            with open(file) as f:
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

    def padding_words(self) -> None:
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
        d = [[self.word2index[w] for w in s] for s in self.sentences]
        split = int(len(d) * ratio)
        np.random.seed(seed)
        np.random.shuffle(d)
        np.random.seed(seed)
        np.random.shuffle(self.labels)
        self.x_train, self.x_test = (np.array(d[:split])[:, np.newaxis, :],
                                     np.array(d[split:])[:, np.newaxis, :])
        self.y_train, self.y_test = (np.array(self.labels[:split]),
                                     np.array(self.labels[split:]))

        return
    
    def embed(self) -> "numpy.ndarray":
        """
        Embedding by gensim.Word2Vec skip-gram model and extract weight vector.
        """
        print("Training Word2vec model...")
        s = [[self.word2index[w] for w in s] for s in self.sentences]
        self.embed_model = Word2Vec(s, workers=self.w2v_num_workers,
                                    size=self.w2v_vectorize_dim,
                                    min_count=self.w2v_min_word_count,
                                    window=self.w2v_context,
                                    sample=self.w2v_downsampling)
        self.embed_model.init_sims(replace=True)
        self.embed_weights = {key: self.embed_model[word] if word in self.embed_model
            else np.random.uniform(-0.25, 0.25, self.embed_model.vector_size)
            for key, word in self.word2index.items()}
                                                                                
        print("got embedding weights!")
        return self.embed_weights        


if __name__ == "__main__":
    pass
