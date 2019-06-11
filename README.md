# CNN-with-CIFAR-10
Objective: Study the impact of loss function and activation function on the CNN
 
A 3-layer [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) with three activation functions was designed and used to classify the images in the well-known [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) set, and the outcome was compared with the expected results for CIFAR-10.

Activation Function:

Out of the three common methods, namely Sigmoid, ReLU and ELU, the latter was used in this project because the first two methods have the following downsides:
1- [Sigmiod](https://en.wikipedia.org/wiki/Sigmoid_function) has high fluctuation and vanishing gradainet, 
2- [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) is linear for all positive values, but zero for negative ones,
3- [ELU](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html) has a small slopeSGD for negative values, which addresses the downsides of the previous funcitons.

Optimization Methods:

SGD, Adam, Adagrad and RMSProp were used for optimization, out of which SGD had the lowest accuracy.

Results:

As seen in the results, the [Dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network) layer prevents the network from overfitting.
