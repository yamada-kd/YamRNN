# YamRNN
**Ya**mada **m**odified **RNN**

## Abstract
Recurrent neural networks (RNNs) are among the most promising of the many artificial intelligence techniques now under development, showing great potential for memory, interaction, and linguistic understanding. Among the more sophisticated RNNs are long short-term memory (LSTM) and gated recurrent units (GRUs), which emulate animal brain behavior; these methods yield superior memory and learning speed because of the excellent core structure of their architectures. In this study, we attempted to make further improvements in core structure and develop a novel, compact architecture with a high learning speed. We stochastically generated 30,000 RNN architectures, evaluated their performance, and selected the one most capable of memorizing long contexts with relatively few parameters. This RNN, YamRNN, had fewer parameters than LSTM and GRU by a factor of two-thirds or better and reduced the time required to achieve the same learning performance on a sequence classification task as LSTM and GRU by 80% at maximum. This novel RNN architecture is expected to be useful for addressing problems such as predictions and analyses on contextual data and also suggests that there is room for the development of better architectures.

## Implementation of RNNs
### YamRNN
Paper: [Developing a novel recurrent neural network architecture with fewer parameters and good learning performance](https://www.biorxiv.org/content/10.1101/2020.04.08.031484v1)

Code for Tensorflow: [code/tensorflow/yamrnn.py](https://github.com/yamada-kd/YamRNN/blob/master/code/tensorflow/yamrnn.py)

Code for Theano: [code/theano/yamrnn.py](https://github.com/yamada-kd/YamRNN/blob/master/code/theano/yamrnn.py)

### Minimal Gated Unit (MGU)
Paper: [Minimal gated unit for recurrent neural networks](https://link.springer.com/article/10.1007/s11633-016-1006-2)

Code for Tensorflow: [code/tensorflow/mgu.py](https://github.com/yamada-kd/YamRNN/blob/master/code/tensorflow/mgu.py)

Code for Theano: [code/theano/mgu.py](https://github.com/yamada-kd/YamRNN/blob/master/code/theano/mgu.py)

### Simplified LSTM (S-LSTM)
Paper: [Investigating gated recurrent neural networks for speech synthesis](https://arxiv.org/abs/1601.02539)

Code for Tensorflow: [code/tensorflow/slstm.py](https://github.com/yamada-kd/YamRNN/blob/master/code/tensorflow/slstm.py)

Code for Theano: [code/theano/slstm.py](https://github.com/yamada-kd/YamRNN/blob/master/code/theano/slstm.py)
