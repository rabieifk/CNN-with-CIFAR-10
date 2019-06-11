# CNN-with-CIFAR-10
Objective: Study the impact of loss function and activation function on the CNN
 
A 3-layer [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) with three activation functions was designed and used to classify the images in the well-known [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) set, and the outcome was compared with the expected results for CIFAR-10.

Out of three common methods Sigmoid, ReLU and ELU, the latter was used in this project due to the downsides of the other functions; specifically, the first two methods have the following downsides:
** [Sigmiod](https://en.wikipedia.org/wiki/Sigmoid_function) has high fluctuation and vanishing gradainet.
** [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) is linear for all positive values, but zero for negative ones.
** [ELU](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html) consider little slopeSGD for negative values

SGD, Adam, Adagrad and RMSProp are used for optimization. SGD has the lowes accuracy.

As seen in the results, the [Dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network) layer perevents the network from overfitting.
