# CNN-with-CIFAR-10
impact of loss function and activation function on the CNN with CIFAR 10 data set

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset is a useful dataset for analysing the accuracy of classification in many algorithm.

A  three layers [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) in which three activation functions are employed, is designed:

[Sigmiod](https://en.wikipedia.org/wiki/Sigmoid_function) => fluctuation and vanishing gradainet are sigmiod's challenge

[ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) => linear for all positive values is linear, and for all negative values is zero.

[ELU](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html) => consider little slopeSGD for negative values

The best result is for ELU.

SGD, Adam, Adagrad and RMSProp is used for optimization. SGD has the lowes accuracy.

using [Dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network) layer perevents network from overfitting as you can see in the result.
