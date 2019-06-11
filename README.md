# CNN-with-CIFAR-10
Objective: Study the impact of loss function and activation function on the CNN
 
A 3-layer [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) with three activation functions was designed and used to classify the images in the well-known [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) set, and the outcome was compared with the expected results for CIFAR-10.


[Sigmiod](https://en.wikipedia.org/wiki/Sigmoid_function) => fluctuation and vanishing gradainet are sigmiod's challenge

[ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) => linear for all positive values is linear, and for all negative values is zero.

[ELU](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html) => consider little slopeSGD for negative values

The best result is for ELU.

SGD, Adam, Adagrad and RMSProp are used for optimization. SGD has the lowes accuracy.

using [Dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network) layer perevents network from overfitting as you can see in the result.
