# CNN-for-Sentence-Classification-in-Chainer

Implementation of Yoon Kim's [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) with Chainer.

> Abstract (from Cornell university library)
>We report on a series of experiments with convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks. We show that a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks. Learning task-specific vectors through fine-tuning offers further gains in performance. We additionally propose a simple modification to the architecture to allow for the use of both task-specific and static vectors. The CNN models discussed herein improve upon the state of the art on 4 out of 7 tasks, which include sentiment analysis and question classification.

![](./src/structure.png)

- [Yoon Kim's TensorFlow implementation repo](https://github.com/yoonkim/CNN_sentence)
- [Google Prerained Word2Vec model](https://code.google.com/archive/p/word2vec/)
- [skipgram Word2Vec on Chainer example](https://github.com/chainer/chainer/tree/master/examples/word2vec)
## Requirements

- numpy
- pandas
- chainer
- gensim.models Word2vec


## Text Data for Classification

datasets from [cornell dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/)

```
# data location
data/
    |_pos/
    |    |_cv000_01.txt
    |    |_cv000_02.txt
    |      :
    |_neg/
        |_cv000_01.txt
        |_cv000_02.txt
           :
```

## Demo, Usage

```python
import hoge

hoge()
```
